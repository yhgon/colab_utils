import requests
import os
import sys

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

def main(url, filename, target_dir ):
    prefix="https://drive.google.com/file/d/"
    postfix="/view?usp=sharing"
    id=url.replace(prefix, '').replace(postfix, '')
    save_filename=os.path.join(target_dir, filename)
    t = time.time()
    download_file_from_google_drive(id, save_filename)
    elapsed = time.time() - t
    print("it takes {} sec ".format(elapsed) ) 
    
if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--shared_url', required=True, help='Google Drive Shared Link')
    parser.add_argument('-f', '--filename',  type=str, default='download.zip')
    parser.add_argument('-d', '--target_dir',      type=str, default='/content')          
    args = parser.parse_args()
    
    main(args.shared_url, args.filename, args.target_dir )
