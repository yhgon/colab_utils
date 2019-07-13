import requests
import os
import sys
import time

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

def main(url, filename, target_dir ):
    prefix="https://drive.google.com/file/d/"
    postfix="/view?usp=sharing"
    id=url.replace(prefix, '').replace(postfix, '')
    save_filename=os.path.join(target_dir, filename)
    t = time.time()
    download_file_from_google_drive(id, save_filename)
    elapsed = time.time() - t
    print("It took {}sec to download {} {} ".format(round(elapsed, 2) ), file_size(save_filename),  filename) 
    
if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--shared_url', required=True, help='Google Drive Shared Link')
    parser.add_argument('-f', '--filename',  type=str, default='download.zip')
    parser.add_argument('-d', '--target_dir',      type=str, default='./')          
    args = parser.parse_args()
    
    main(args.shared_url, args.filename, args.target_dir )
