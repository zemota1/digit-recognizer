import pandas as pd
import boto3


if __name__ == '__main__':

    config = pd.read_csv("config/key.csv")

    session = boto3.Session(
        aws_access_key_id=str(config['AWSAccessKeyId'][0]),
        aws_secret_access_key=str(config['AWSSecretKey'][0]),
    )

    s3 = session.client('s3')

    df_train = pd.read_csv(
        s3.get_object(Bucket='josemota-mnist-data', Key='train.csv')['Body']
    )

    df_test = pd.read_csv(
        s3.get_object(Bucket='josemota-mnist-data', Key='test.csv')['Body']
    )
