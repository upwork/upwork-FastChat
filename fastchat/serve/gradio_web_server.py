"""
The gradio demo server for chatting with a single model.
"""

import argparse
import base64
from collections import defaultdict
import datetime
import hashlib
import json
import os
import random
import time
import uuid

import fsspec
import gradio as gr
import requests
import yaml

from fastchat.conversation import (
    get_conv_template,
)
from fastchat.constants import (
    LOGDIR,
    WORKER_API_TIMEOUT,
    ErrorCode,
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    RATE_LIMIT_MSG,
    SERVER_ERROR_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SESSION_EXPIRATION_TIME,
)
from fastchat.model.model_adapter import (
    get_conversation_template,
)
from fastchat.model.model_registry import get_model_info, model_info
from fastchat.serve.api_provider import get_api_provider_stream_iter
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.utils import (
    build_logger,
    get_window_url_params_js,
    get_window_url_params_with_tos_js,
    moderation_filter,
    parse_gradio_auth_creds,
    load_image,
)
from queryunderstanding.query_understanding import QueryUnderstanding
from queryunderstanding.utils import load_freelancers, load_job

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "FastChat Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

freelancers: list[dict[str, str]] = load_freelancers()
job: dict[str, str] = load_job()
query_understanding = QueryUnderstanding()

controller_url = None
enable_moderation = False
use_remote_storage = False

acknowledgment_md = """
### Terms of Service

Users are required to agree to the following terms before using the service:

The service is a research preview. It only provides limited safety measures and may generate offensive content.
It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
Please do not upload any private information.
The service collects user dialogue data, including both text and images, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license.

### Acknowledgment
We thank [UC Berkeley SkyLab](https://sky.cs.berkeley.edu/), [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Hyperbolic](https://hyperbolic.xyz/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous [sponsorship](https://lmsys.org/donations/).

<div class="sponsor-image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/skylab.png" alt="SkyLab">
    <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
    <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
    <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
    <img src="https://storage.googleapis.com/public-arena-asset/hyperbolic_logo.png" alt="Hyperbolic">
    <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
    <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
</div>
"""

# JSON file format of API-based models:
# {
#   "gpt-3.5-turbo": {
#     "model_name": "gpt-3.5-turbo",
#     "api_type": "openai",
#     "api_base": "https://api.openai.com/v1",
#     "api_key": "sk-******",
#     "anony_only": false
#   }
# }
#
#  - "api_type" can be one of the following: openai, anthropic, gemini, or mistral. For custom APIs, add a new type and implement it accordingly.
#  - "anony_only" indicates whether to display this model in anonymous mode only.

api_endpoint_info = {}


class State:
    def __init__(self, model_name, is_vision=False):
        model_api_dict = api_endpoint_info.get(model_name, None)
        if model_api_dict and "conv_template" in model_api_dict:
            self.conv = get_conv_template(model_api_dict["conv_template"])
            if "system_template" in model_api_dict:
                self.conv.system_template = model_api_dict["system_template"]
            if "system_message" in model_api_dict:
                self.conv.set_system_message(model_api_dict["system_message"])
        else:
            self.conv = get_conversation_template(model_name)
        self.conv_id = uuid.uuid4().hex
        self.skip_next = False
        self.model_name = model_name
        self.oai_thread_id = None
        self.is_vision = is_vision
        self.assistant_first_message = (
            model_api_dict.get("assistant_first_message", None)
            if model_api_dict
            else None
        )

        # NOTE(chris): This could be sort of a hack since it assumes the user only uploads one image. If they can upload multiple, we should store a list of image hashes.
        self.has_csam_image = False

        self.regen_support = True
        if "browsing" in model_name:
            self.regen_support = False

        system_prompt = self.conv.get_system_message()
        if len(system_prompt) > 0:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            system_prompt = system_prompt.replace("{{currentDateTime}}", current_date)
            self.conv.set_system_message(system_prompt)

    def to_gradio_chatbot(self):
        history = self.conv.to_gradio_chatbot()

        if self.assistant_first_message is not None:
            history = [(None, self.assistant_first_message)] + history
        return history

    def dict(self):
        base = self.conv.dict()
        base.update(
            {
                "conv_id": self.conv_id,
                "model_name": self.model_name,
                "has_csam_image": self.has_csam_image,
            }
        )
        return base


def set_global_vars(controller_url_, enable_moderation_, use_remote_storage_):
    global controller_url, enable_moderation, use_remote_storage
    controller_url = controller_url_
    enable_moderation = enable_moderation_
    use_remote_storage = use_remote_storage_


def get_conv_log_filename(is_vision=False):
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
    if is_vision:
        name = os.path.join(LOGDIR, f"vision-tmp-{conv_log_filename}")
    else:
        name = os.path.join(LOGDIR, conv_log_filename)

    return name


def get_model_list(controller_url, register_api_endpoint_file, vision_arena):
    global api_endpoint_info

    # Add models from the controller
    if controller_url:
        ret = requests.post(controller_url + "/refresh_all_workers")
        assert ret.status_code == 200

        if vision_arena:
            ret = requests.post(controller_url + "/list_multimodal_models")
            models = ret.json()["models"]
        else:
            ret = requests.post(controller_url + "/list_language_models")
            models = ret.json()["models"]
    else:
        models = []

    # Add models from the API providers
    if register_api_endpoint_file:
        fs, fspath = fsspec.url_to_fs(register_api_endpoint_file)
        try:
            with fs.open(fspath, "r") as file:
                api_endpoint_info = json.loads(file.read())
        except (FileNotFoundError, json.decoder.JSONDecodeError) as err:
            api_endpoint_info = {
                "error-loading-api-endpoints": {
                    "model_name": "gpt-3.5-turbo",
                    "api_type": "openai",
                    "assistant_first_message": f"Error loading api endpoint file: {err}",
                }
            }
        for mdl, mdl_dict in api_endpoint_info.items():
            mdl_vision = mdl_dict.get("vision-arena", False)
            mdl_text = mdl_dict.get("text-arena", True)
            if vision_arena and mdl_vision:
                models.append(mdl)
            if not vision_arena and mdl_text:
                models.append(mdl)

    # Remove anonymous models
    models = list(set(models))
    visible_models = models.copy()
    for mdl in models:
        if mdl not in api_endpoint_info:
            continue
        mdl_dict = api_endpoint_info[mdl]
        if mdl_dict.get("anony_only", False):
            visible_models.remove(mdl)

    # Sort models and add descriptions
    priority = {k: f"___{i:03d}" for i, k in enumerate(model_info)}
    models.sort(key=lambda x: priority.get(x, x))
    visible_models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"All models: {models}")
    logger.info(f"Visible models: {visible_models}")
    return visible_models, models


def load_demo_single(models, url_params):
    selected_model = models[0] if len(models) > 0 else ""
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            selected_model = model

    dropdown_update = gr.Dropdown(choices=models, value=selected_model, visible=True)
    return dropdown_update


def load_demo(url_params, request: gr.Request):
    global models

    ip = get_ip(request)
    logger.info(f"load_demo. ip: {ip}. params: {url_params}")

    if args.model_list_mode == "reload":
        models, all_models = get_model_list(
            controller_url, args.register_api_endpoint_file, vision_arena=False
        )

    return load_demo_single(models, url_params), url_params


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    filename = get_conv_log_filename()
    if "llava" in model_selector:
        filename = filename.replace("2024", "vision-tmp-2024")

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def upvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def undo(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"undo. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 6
    # Remove all assistant responses before the last user message.
    while state.conv.messages and state.conv.messages[-1][0] != state.conv.roles[0]:
        state.conv.messages.pop()
    # Remove the last user message too.
    if state.conv.messages:
        state.conv.messages.pop()
    if not state.conv.messages:
        new_btn_state = disable_btn
    else:
        new_btn_state = no_change_btn
    return (state, state.to_gradio_chatbot(), "", None) + (new_btn_state,) * 6


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 6
    # Remove all assistant responses before the last user message.
    while state.conv.messages and state.conv.messages[-1][0] != state.conv.roles[0]:
        state.conv.messages.pop()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 6


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
    else:
        ip = request.client.host
    return ip


# TODO(Chris): At some point, we would like this to be a live-reporting feature.
def report_csam_image(state, image):
    pass


def _prepare_text_with_image(state, text, images, csam_flag):
    if images is not None and len(images) > 0:
        image = images[0]

        if len(state.conv.get_images()) > 0:
            # reset convo with new image
            state.conv = get_conversation_template(state.model_name)

        resize_image = "llava" in state.model_name
        image = state.conv.convert_image_to_base64(
            image,
            resize_image=resize_image,
        )  # PIL type is not JSON serializable

        if csam_flag:
            state.has_csam_image = True
            report_csam_image(state, image)

        text = text, [image]

    return text


def get_state(state, model_selector, example_selector):
    return base64.urlsafe_b64encode(
        json.dumps(
            {
                "model": model_selector,
                "example": example_selector,
                "system_message": state.conv.get_system_message(),
                "messages": state.conv.messages,
            }
        ).encode("utf-8")
    ).decode("utf-8")


def clear_system_message():
    return ""


def init_chat(
    request: gr.Request,
    state,
    model_selector,
    example_selector,
    system_message,
    url_params=None,
):
    ip = get_ip(request)
    no_examples_label = "Examples..."

    url_state = {}
    if url_params and "state" in url_params:
        state_str = url_params["state"].encode("utf-8")
        url_state = json.loads(base64.urlsafe_b64decode(state_str).decode("utf-8"))
        model_selector = url_state.get("model", "")
        example_selector = url_state.get("example", "")
    else:
        if not state or state.model_name != model_selector:
            example_selector = no_examples_label
    logger.info(f"init_chat. ip: {ip} state: {url_state}")

    state = State(model_selector)
    new_system_message = None
    if not system_message:
        new_system_message = state.conv.get_system_message()
    examples_dict = {}
    model_api_dict = api_endpoint_info.get(model_selector, None)
    if model_api_dict:
        examples_file = model_api_dict.get("examples_file", None)
        if examples_file:
            fs, fspath = fsspec.url_to_fs(examples_file)
            try:
                with fs.open(fspath, "r") as file:
                    if fspath.endswith(".yaml"):
                        examples_dict = yaml.safe_load(file.read())
                    else:
                        examples_dict = json.loads(file.read())
            except (FileNotFoundError, json.decoder.JSONDecodeError) as err:
                examples_dict = {
                    "error-loading-examples": [
                        ["assistant", f"Error loading examples file: {err}"]
                    ]
                }
                example_selector = "error-loading-examples"
    if example_selector and example_selector in examples_dict:
        messages = examples_dict[example_selector]
        if not isinstance(messages, list):
            raise ValueError("Examples must be a list of messages.")
        for i, turn in enumerate(messages):
            if isinstance(turn, dict):
                messages[i] = (turn["role"], turn["content"])
        state.conv.messages = messages
        if state.conv.messages and state.conv.messages[0][0] == "system":
            if not system_message:
                new_system_message = state.conv.messages[0][1]
            state.conv.messages = state.conv.messages[1:]
        state.assistant_first_message = None

    if url_state:
        new_system_message = url_state.get("system_message", "")
    if new_system_message is not None:
        system_message = new_system_message
    state.conv.set_system_message(system_message)
    if url_state:
        state.conv.messages = url_state.get("messages", [])
        state.assistant_first_message = None

    model_selector_dropdown = gr.Dropdown(
        value=model_selector,
    )
    example_selector_dropdown = gr.Dropdown(
        choices=[no_examples_label] + list(examples_dict.keys()),
        value=example_selector,
        visible=bool(examples_dict),
    )
    return (
        state,
        state.to_gradio_chatbot(),
        system_message,
        "",
        None,
        model_selector_dropdown,
        example_selector_dropdown,
    ) + (disable_btn,) * 6


def add_text(state, model_selector, text, image, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 6

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    # flagged = moderation_filter(text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (
            state,
            state.to_gradio_chatbot(),
            CONVERSATION_LIMIT_MSG,
            None,
        ) + (no_change_btn,) * 6

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    text = _prepare_text_with_image(state, text, image, csam_flag=False)
    state.conv.append_message(state.conv.roles[0], text)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 6


def model_worker_stream_iter(
    conv,
    model_name,
    worker_addr,
    prompt,
    temperature,
    repetition_penalty,
    top_p,
    max_new_tokens,
    images,
):
    # Make requests
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    logger.info(f"==== request ====\n{gen_params}")

    if len(images) > 0:
        gen_params["images"] = images

    # Stream output
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json=gen_params,
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def is_limit_reached(model_name, ip):
    monitor_url = "http://localhost:9090"
    try:
        ret = requests.get(
            f"{monitor_url}/is_limit_reached?model={model_name}&user_id={ip}", timeout=1
        )
        obj = ret.json()
        return obj
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        logger.info(f"monitor error: {type(e)} {e}")
        return None


def generate_turn(
    state,
    role,
    temperature,
    top_p,
    max_new_tokens,
    request: gr.Request,
    use_recommended_config=False,
    rag=None,
    summarize_results=False,
):
    start_tstamp = time.time()
    conv, model_name = state.conv, state.model_name
    conv.append_message(role, None)
    model_api_dict = api_endpoint_info.get(model_name, None)
    images = conv.get_images()

    if rag:
        retrieved_context = query_understanding.search(
            conv,
            freelancers=freelancers,
            job=job,
            enforce_rag=rag,
            summarize_results=summarize_results,
        )
        conv.update_last_message(retrieved_context)
        return

    if model_api_dict is None:
        # Query worker address
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        worker_addr = ret.json()["address"]
        logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

        # No available worker
        if worker_addr == "":
            conv.update_last_message(SERVER_ERROR_MSG)
            yield (
                state,
                state.to_gradio_chatbot(),
                disable_btn,
                disable_btn,
                disable_btn,
                enable_btn,
                enable_btn,
                enable_btn,
            )
            return

        # Construct prompt.
        # We need to call it here, so it will not be affected by "▌".
        prompt = conv.get_prompt()
        # Set repetition_penalty
        if "t5" in model_name:
            repetition_penalty = 1.2
        else:
            repetition_penalty = 1.0

        stream_iter = model_worker_stream_iter(
            conv,
            model_name,
            worker_addr,
            prompt,
            temperature,
            repetition_penalty,
            top_p,
            max_new_tokens,
            images,
        )
    else:
        stream = True
        if use_recommended_config:
            recommended_config = model_api_dict.get("recommended_config", None)
            if recommended_config is not None:
                temperature = recommended_config.get("temperature", temperature)
                top_p = recommended_config.get("top_p", top_p)
                max_new_tokens = recommended_config.get(
                    "max_new_tokens", max_new_tokens
                )
        config_override = model_api_dict.get("config_override", None)
        if config_override is not None:
            temperature = config_override.get("temperature", temperature)
            top_p = config_override.get("top_p", top_p)
            max_new_tokens = config_override.get("max_new_tokens", max_new_tokens)
            stream = config_override.get("stream", stream)

        stream_iter = get_api_provider_stream_iter(
            conv,
            model_name,
            model_api_dict,
            temperature,
            top_p,
            max_new_tokens,
            state,
            stream=stream,
        )

    # html_code = ' <span class="cursor"></span> '
    html_code = "▌"

    # conv.update_last_message("▌")
    conv.update_last_message(html_code)
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 6

    try:
        data = {"text": ""}
        for i, data in enumerate(stream_iter):
            if data["error_code"] == 0:
                output = data["text"].strip()
                # conv.update_last_message(output + "▌")
                conv.update_last_message(output + html_code)
                yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 6
            else:
                output = data["text"] + f"\n\n(error_code: {data['error_code']})"
                conv.update_last_message(output)
                yield (state, state.to_gradio_chatbot()) + (
                    disable_btn,
                    disable_btn,
                    disable_btn,
                    enable_btn,
                    enable_btn,
                    enable_btn,
                )
                return
        output = data["text"].strip()
        conv.update_last_message(output)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 6
    except requests.exceptions.RequestException as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_REQUEST_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
            enable_btn,
        )
        return
    except Exception as e:
        conv.update_last_message(
            f"{SERVER_ERROR_MSG}\n\n"
            f"(error_code: {ErrorCode.GRADIO_STREAM_UNKNOWN_ERROR}, {e})"
        )
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
            enable_btn,
        )
        return

    finish_tstamp = time.time()
    logger.info(f"{output}")

    conv.save_new_images(
        has_csam_images=state.has_csam_image, use_remote_storage=use_remote_storage
    )

    filename = get_conv_log_filename(is_vision=state.is_vision)

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def bot_response(
    state,
    temperature,
    top_p,
    max_new_tokens,
    generate_thoughts,
    rag,
    summarize_results,
    request: gr.Request,
    apply_rate_limit=True,
    use_recommended_config=False,
):
    ip = get_ip(request)
    logger.info(f"bot_response. ip: {ip}")
    temperature = float(temperature)
    top_p = float(top_p)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        state.skip_next = False
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 6
        return

    if apply_rate_limit:
        ret = is_limit_reached(state.model_name, ip)
        if ret is not None and ret["is_limit_reached"]:
            error_msg = RATE_LIMIT_MSG + "\n\n" + ret["reason"]
            logger.info(f"rate limit reached. ip: {ip}. error_msg: {ret['reason']}")
            state.conv.update_last_message(error_msg)
            yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 6
            return

    logger.info(f"RAG: {rag}")
    if rag != "No RAG":
        yield from generate_turn(
            state,
            role=state.conv.roles[3],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            request=request,
            use_recommended_config=use_recommended_config,
            rag=rag,
            summarize_results=summarize_results,
        )

    model_api_dict = api_endpoint_info.get(state.model_name, None)
    if (
        generate_thoughts
        and model_api_dict is not None
        and model_api_dict.get("has_thoughts", False)
    ):
        # Generate thoughts turn.
        yield from generate_turn(
            state,
            role=state.conv.roles[2],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            request=request,
            use_recommended_config=use_recommended_config,
        )

    yield from generate_turn(
        state,
        role=state.conv.roles[1],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        request=request,
        use_recommended_config=use_recommended_config,
    )


block_css = """
#notice_markdown .prose {
    font-size: 110% !important;
}
#notice_markdown th {
    display: none;
}
#notice_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#arena_leaderboard_dataframe table {
    font-size: 110%;
}
#full_leaderboard_dataframe table {
    font-size: 110%;
}
#model_description_markdown {
    font-size: 110% !important;
}
#leaderboard_markdown .prose {
    font-size: 110% !important;
}
#leaderboard_markdown td {
    padding-top: 6px;
    padding-bottom: 6px;
}
#leaderboard_dataframe td {
    line-height: 0.1em;
}
#about_markdown .prose {
    font-size: 110% !important;
}
#ack_markdown .prose {
    font-size: 110% !important;
}
#chatbot .prose {
    font-size: 105% !important;
}
.sponsor-image-about img {
    margin: 0 20px;
    margin-top: 20px;
    height: 40px;
    max-height: 100%;
    width: auto;
    float: left;
}

.chatbot h1, h2, h3 {
    margin-top: 8px; /* Adjust the value as needed */
    margin-bottom: 0px; /* Adjust the value as needed */
    padding-bottom: 0px;
}

.chatbot h1 {
    font-size: 130%;
}
.chatbot h2 {
    font-size: 120%;
}
.chatbot h3 {
    font-size: 110%;
}
.chatbot p:not(:first-child) {
    margin-top: 8px;
}

.typing {
    display: inline-block;
}

.cursor {
    display: inline-block;
    width: 7px;
    height: 1em;
    background-color: black;
    vertical-align: middle;
    animation: blink 1s infinite;
}

.dark .cursor {
    display: inline-block;
    width: 7px;
    height: 1em;
    background-color: white;
    vertical-align: middle;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    50.1%, 100% { opacity: 0; }
}

.app {
  max-width: 100% !important;
  padding: 20px !important;
}

a {
    color: #1976D2; /* Your current link color, a shade of blue */
    text-decoration: none; /* Removes underline from links */
}
a:hover {
    color: #63A4FF; /* This can be any color you choose for hover */
    text-decoration: underline; /* Adds underline on hover */
}
"""


def build_about():
    about_markdown = """
# About Us
Chatbot Arena is an open-source research project developed by members from [LMSYS](https://lmsys.org) and UC Berkeley [SkyLab](https://sky.cs.berkeley.edu/). Our mission is to build an open platform to evaluate LLMs by human preference in the real-world.
We open-source our [FastChat](https://github.com/lm-sys/FastChat) project at GitHub and release chat and human feedback dataset. We invite everyone to join us!

## Arena Core Team
- [Lianmin Zheng](https://lmzheng.net/) (co-lead), [Wei-Lin Chiang](https://infwinston.github.io/) (co-lead), [Ying Sheng](https://sites.google.com/view/yingsheng/home), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Ion Stoica](http://people.eecs.berkeley.edu/~istoica/)

## Past Members
- [Siyuan Zhuang](https://scholar.google.com/citations?user=KSZmI5EAAAAJ), [Hao Zhang](https://cseweb.ucsd.edu/~haozhang/)

## Learn more
- Chatbot Arena [paper](https://arxiv.org/abs/2403.04132), [launch blog](https://lmsys.org/blog/2023-05-03-arena/), [dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md), [policy](https://lmsys.org/blog/2024-03-01-policy/)
- LMSYS-Chat-1M dataset [paper](https://arxiv.org/abs/2309.11998), LLM Judge [paper](https://arxiv.org/abs/2306.05685)

## Contact Us
- Follow our [X](https://x.com/lmsysorg), [Discord](https://discord.gg/HSWAKCrnFx) or email us at lmsys.org@gmail.com
- File issues on [GitHub](https://github.com/lm-sys/FastChat)
- Download our datasets and models on [HuggingFace](https://huggingface.co/lmsys)

## Acknowledgment
We thank [SkyPilot](https://github.com/skypilot-org/skypilot) and [Gradio](https://github.com/gradio-app/gradio) team for their system support.
We also thank [UC Berkeley SkyLab](https://sky.cs.berkeley.edu/), [Kaggle](https://www.kaggle.com/), [MBZUAI](https://mbzuai.ac.ae/), [a16z](https://www.a16z.com/), [Together AI](https://www.together.ai/), [Hyperbolic](https://hyperbolic.xyz/), [Anyscale](https://www.anyscale.com/), [HuggingFace](https://huggingface.co/) for their generous sponsorship. Learn more about partnership [here](https://lmsys.org/donations/).

<div class="sponsor-image-about">
    <img src="https://storage.googleapis.com/public-arena-asset/skylab.png" alt="SkyLab">
    <img src="https://storage.googleapis.com/public-arena-asset/kaggle.png" alt="Kaggle">
    <img src="https://storage.googleapis.com/public-arena-asset/mbzuai.jpeg" alt="MBZUAI">
    <img src="https://storage.googleapis.com/public-arena-asset/a16z.jpeg" alt="a16z">
    <img src="https://storage.googleapis.com/public-arena-asset/together.png" alt="Together AI">
    <img src="https://storage.googleapis.com/public-arena-asset/hyperbolic_logo.png" alt="Hyperbolic">
    <img src="https://storage.googleapis.com/public-arena-asset/anyscale.png" alt="AnyScale">
    <img src="https://storage.googleapis.com/public-arena-asset/huggingface.png" alt="HuggingFace">
</div>
"""
    gr.Markdown(about_markdown, elem_id="about_markdown")


def build_single_model_ui(demo, models, add_promotion_links=False, add_load_demo=True):
    promotion = (
        """
    - | [GitHub](https://github.com/lm-sys/FastChat) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |
    - Introducing Llama 2: The Next Generation Open Source Large Language Model. [[Website]](https://ai.meta.com/llama/)
    - Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality. [[Blog]](https://lmsys.org/blog/2023-03-30-vicuna/)

    ## 🤖 Choose any model to chat
    """
        if add_promotion_links
        else ""
    )

    notice_markdown = f"""
    # 🏔️ Chat with Open Large Language Models
    {promotion}
    """

    state = gr.State()
    share_str = gr.Textbox(visible=False)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Accordion("📄 Job Information", open=True):
        job_info_html = f"""
        <div style="padding: 10px;">
            <h3>{job['title']}</h3>
            <p>{job['description']}</p>
        </div>
        """
        gr.HTML(job_info_html, elem_id="job_info")

    with gr.Accordion("👥 Freelancers", open=True):
        freelancers_table = """
        <table style="width:100%; border-collapse: collapse;">
            <tr>
                <th style="text-align: left; padding: 8px;">Name</th>
                <th style="text-align: left; padding: 8px;">Title</th>
                <th style="text-align: left; padding: 8px;">Profile</th>
            </tr>
        """
        for freelancer in freelancers:
            freelancers_table += f"""
            <tr>
                <td style="padding: 8px;">{freelancer['name']}</td>
                <td style="padding: 8px;">{freelancer['title']}</td>
                <td style="padding: 8px;"><a href="{freelancer['url']}" target="_blank">View Profile</a></td>
            </tr>
            """
        freelancers_table += "</table>"
        gr.HTML(freelancers_table, elem_id="freelancer_list")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row(elem_id="model_selector_row"):
            model_selector = gr.Dropdown(
                choices=models,
                value=models[0] if len(models) > 0 else "",
                interactive=True,
                show_label=False,
                container=False,
            )
            example_selector = gr.Dropdown(
                visible=False,
                interactive=True,
                show_label=False,
                container=False,
            )
        with gr.Row(elem_id="rag_selector_row"):
            rag_selector = gr.Dropdown(
                choices=[
                    "No RAG",
                    "Context-Aware",
                    "Hybrid",
                    "Vector Search",
                    "Knowledge Graph",
                ],
                value="Context-Aware",
                interactive=True,
                show_label=False,
                container=False,
            )
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Scroll down and start chatting",
            height=550,
            show_copy_button=True,
            render_markdown=False,
        )
    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False, visible=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False, visible=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False, visible=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        undo_btn = gr.Button(value="⟲  Undo", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        share_btn = gr.Button(value="⎘ Copy Link")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        system_message = gr.Textbox(
            label="System Message",
            lines=6,
        )
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=64,
            maximum=8192,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )
        generate_thoughts = gr.Checkbox(value=True, label="Generate thoughts")
        summarize_results = gr.Checkbox(value=False, label="Summarize results")

    if add_promotion_links:
        gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    imagebox = gr.State(None)
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, undo_btn, clear_btn]

    upvote_btn.click(
        upvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    downvote_btn.click(
        downvote_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    flag_btn.click(
        flag_last_response,
        [state, model_selector],
        [textbox, upvote_btn, downvote_btn, flag_btn],
    )
    undo_btn.click(undo, state, [state, chatbot, textbox, imagebox] + btn_list)
    regenerate_btn.click(
        regenerate, state, [state, chatbot, textbox, imagebox] + btn_list
    ).then(
        bot_response,
        [
            state,
            temperature,
            top_p,
            max_output_tokens,
            generate_thoughts,
            summarize_results,
            rag_selector,
        ],
        [state, chatbot] + btn_list,
    )

    share_btn.click(
        get_state,
        [state, model_selector, example_selector],
        [share_str],
    ).then(
        None,
        [share_str],
        js=r"""
function copy(share_str) {
  let text = window.location.href.split('?')[0] + '?state=' + share_str;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text);
  } else {
    const textarea = document.createElement('textarea');
    textarea.textContent = text;
    document.body.append(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
  }
}
""",
    )

    gr.on(
        [clear_btn.click, system_message.blur],
        init_chat,
        [state, model_selector, example_selector, system_message],
        [
            state,
            chatbot,
            system_message,
            textbox,
            imagebox,
            model_selector,
            example_selector,
        ]
        + btn_list,
    )
    gr.on(
        [model_selector.input, example_selector.input],
        clear_system_message,
        [],
        [system_message],
    ).then(
        init_chat,
        [state, model_selector, example_selector, system_message],
        [
            state,
            chatbot,
            system_message,
            textbox,
            imagebox,
            model_selector,
            example_selector,
        ]
        + btn_list,
    )

    gr.on(
        [textbox.submit, send_btn.click],
        add_text,
        [state, model_selector, textbox, imagebox],
        [state, chatbot, textbox, imagebox] + btn_list,
    ).then(
        bot_response,
        [
            state,
            temperature,
            top_p,
            max_output_tokens,
            generate_thoughts,
            rag_selector,
            summarize_results,
        ],
        [state, chatbot] + btn_list,
    )

    url_params = gr.JSON(visible=False)
    if args.show_terms_of_use:
        load_js = get_window_url_params_with_tos_js
    else:
        load_js = get_window_url_params_js

    if add_load_demo:
        demo.load(
            load_demo,
            [url_params],
            [model_selector, url_params],
            js=load_js,
        ).then(
            init_chat,
            [state, model_selector, example_selector, system_message, url_params],
            [
                state,
                chatbot,
                system_message,
                textbox,
                imagebox,
                model_selector,
                example_selector,
            ]
            + btn_list,
        )

    return [state, model_selector]


def build_demo(models):
    if args.model_list_mode not in ["once", "reload"]:
        raise ValueError(f"Unknown model list mode: {args.model_list_mode}")
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        build_single_model_ui(demo, models, add_load_demo=True)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Whether to generate a public, shareable link",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:21001",
        help="The address of the controller",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=10,
        help="The concurrency count of the gradio queue",
    )
    parser.add_argument(
        "--model-list-mode",
        type=str,
        default="once",
        choices=["once", "reload"],
        help="Whether to load the model list once or reload the model list every time",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Enable content moderation to block unsafe inputs",
    )
    parser.add_argument(
        "--show-terms-of-use",
        action="store_true",
        help="Shows term of use before loading the demo",
    )
    parser.add_argument(
        "--register-api-endpoint-file",
        type=str,
        help="Register API-based model endpoints from a JSON file",
    )
    parser.add_argument(
        "--gradio-auth-path",
        type=str,
        help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"',
    )
    parser.add_argument(
        "--gradio-root-path",
        type=str,
        help="Sets the gradio root path, eg /abc/def. Useful when running behind a reverse-proxy or at a custom URL path prefix",
    )
    parser.add_argument(
        "--use-remote-storage",
        action="store_true",
        default=False,
        help="Uploads image files to google cloud storage if set to true",
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Set global variables
    set_global_vars(args.controller_url, args.moderate, args.use_remote_storage)
    models, all_models = get_model_list(
        args.controller_url, args.register_api_endpoint_file, vision_arena=False
    )

    # Set authorization credentials
    auth = None
    if args.gradio_auth_path is not None:
        auth = parse_gradio_auth_creds(args.gradio_auth_path)

    # Launch the demo
    demo = build_demo(models)
    demo.queue(
        default_concurrency_limit=args.concurrency_count,
        status_update_rate=10,
        api_open=False,
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=200,
        auth=auth,
        root_path=args.gradio_root_path,
    )
