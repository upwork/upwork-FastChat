import os
import boto3 
import logging
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Load environment variables from .env file
load_dotenv()

logging.info("Loading configuration from environment variables and .env file")

# Load environment variables
NEPTUNE_ENDPOINT = os.getenv('NEPTUNE_ENDPOINT')
NEPTUNE_PORT = os.getenv('NEPTUNE_PORT')
AWS_REGION = os.getenv('AWS_REGION')
AWS_PROFILE = os.getenv('AWS_PROFILE', 'up_ml')  # If you want to use a specific profile

LLM_MODEL = os.getenv('LLM_MODEL')
TIMEOUT_MS = int(os.getenv('TIMEOUT_MS', 7200000))

logging.info(f"Loaded Neptune endpoint: {NEPTUNE_ENDPOINT}")
logging.info(f"Loaded timeout value: {TIMEOUT_MS} ms")

# Pagination settings
#VOLUME_THRESHOLD = 100000000000000000000000  # Volume threshold for adding pagination
MAX_LIMIT = 5000  # Number of results per page for pagination

# Example volume table mapping node labels to their respective volumes
NODE_VOLUME_TABLE = {
        'Proposal':	138000000,
        'Contract':	27375095,
        'Client':	23267863,
        'Freelancer':	82453370,
        'SpecializedProfile':	55428783,
        'jobPost':	17225132,
        'WorkHistory':	24011842,
        'Offer':	30468404,
        'Country':	253,
        'City':	5046923,
        'Skill':	14042,
        'Category':	280,
        'AG':	722,
        }

# Schema definition (example)
SCHEMA =""" 
                Contract-[has_freelancer]->Freelancer
                Contract-[created_for_opening]->jobPost
                Contract-[has_client]->Client
                Client-[has_posted]->jobPost
                Freelancer-[has_profile]->SpecializedProfile
                Freelancer-[submitted_proposal]->Proposal
                Freelancer-[interviewed_for]->jobPost
                Freelancer-[hired_for]->jobPost
                Freelancer-[belong_to_category]->Category
                Freelancer-[applied_to]->jobPost
                Freelancer-[has_asserted_skill]->Skill
                SpecializedProfile-[has_category]->Category
                SpecializedProfile-[has_asserted_skill]->Skill
                SpecializedProfile-[has_attribute_group]->AG
                jobPost-[requires_category]->Category
                jobPost-[requires_skill]->Skill
                WorkHistory-[related_job_post]->jobPost
                WorkHistory-[for_freelancer]->Freelancer
                WorkHistory-[for_client]->Client
                Offer-[based_on_job]->jobPost
                Offer-[for_freelancer]->Freelancer
                Offer-[from_client]->Client
                Offer-[based_on_contract]->Contract
                Proposal-[proposal_for]->jobPost
                Proposal-[has_category]->Category
                Category-[has_parent]->Category
                Category-[has_child]->Category
                Category-[related_ag]->AG
                Skill-[belongs_to_ag]->AG
                
                Node properties:
                        "uid": 844465040
                        "status": 731669589
                        "cover_letter": 592707090
                        "bid": 592689435
                        "experience_tier": 263809246
                        "created_ts": 236015057
                        "is_invited": 236015057
                        "modified_ts": 236015057
                        "recommended_score": 236015057
                        "title": 226227257
                        "description": 140210286
                          "ctime": 111606279
                          "hourly_rate": 111168310
                          "type": 86016971
                          "accounting_entity": 83783253
                          "gsv_total": 83783253
                          "hours_per_week": 83783253
                          "is_available": 83783253
                          "jss": 83783253
                          "qscore": 83783253
                          "top_rated_plus_status": 83783253
                          "top_rated_status": 83783253
                          "verification_status": 83783253
                          "creation_date": 83782719
                          "rate": 56427033
                          "amount": 55179246
                          "is_accepted": 30837725
                          "offer_date": 30837725
                          "offer_status": 30837725
                          "company_uid": 27794189
                          "engagement_type": 27794189
                          "expected_duration": 27794189
                          "max_hourly_rate": 27794189
                          "min_hourly_rate": 27794189
                          "mtime": 27794189
                          "no_of_freelancers": 27794189
                          "organization_uid": 27794189
                          "delivery_model": 27385057
                          "start_date": 27385057
                          "client_availability_score": 24205950
                          "client_comment": 24205950
                          "client_communication_score": 24205950
                          "client_cooperation_score": 24205950
                          "client_deadlines_score": 24205950
                          "client_quality_score": 24205950
                          "client_skills_score": 24205950
                          "client_total_score": 24205950
                          "freelancer_availability_score": 24205950
                          "freelancer_comment": 24205950
                          "freelancer_communication_score": 24205950
                          "freelancer_cooperation_score": 24205950
                          "freelancer_deadlines_score": 24205950
                          "freelancer_quality_score": 24205950
                          "freelancer_skills_score": 24205950
                          "freelancer_total_score": 24205950
                          "accountingentity": 23630420
                          "companyID": 23630420
                          "company_size": 23630420
                          "pref_label": 5078658
                          "alt_label": 5063830
                          "state_code": 5063830
                          "timezone": 5063830
                          "entity_status": 15045
                          "ontology_id": 15045
                          "code3": 506
                          "iso_code": 506
                          "taxonomy_level": 280
"""

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

if aws_access_key_id and aws_secret_access_key:
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    logging.info("AWS session created successfully")
else:
    logging.warning("AWS credentials are missing or improperly configured.")
