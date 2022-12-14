import os
import boto3
import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")
# comment out before commit
# from dotenv import load_dotenv

# load_dotenv("./.env")

def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    logger.info(
        "Env vars within download models helper are: {}, {}, {}".format(
            os.environ.get("AWS_ROLE_ARN"),
            os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"),
            os.environ.get("AWS_DEFAULT_REGION"),
        )
    )
    logger.info(
        "Model files will be saved in: {}".format(
            os.path.join(os.getcwd(), 'models')
        )
    )

    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in result.get('Contents', []):
            path, filename = os.path.split(file.get('Key'))
            path = path.replace('data_science/ai_model_files/content-captioning/', local)
            #dest_pathname = os.path.join(local, file.get('Key'))
            dest_pathname = os.path.join(path, filename)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
                resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)

def download_models_helper(models_path = 'models/'):
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    download_dir(client, resource, 'data_science/ai_model_files/content-captioning/version_1/', models_path, bucket=os.environ.get("AWS_BUCKET_NAME"))

# download_models_helper()