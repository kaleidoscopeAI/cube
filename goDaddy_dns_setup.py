import requests

GODADDY_API_KEY = "your_api_key"
GODADDY_API_SECRET = "your_api_secret"
DOMAIN = "artificialthinker.com"

def update_dns(server_ip):
    headers = {"Authorization": f"sso-key {GODADDY_API_KEY}:{GODADDY_API_SECRET}"}
    dns_data = [{"data": server_ip, "ttl": 600}]
    url = f"https://api.godaddy.com/v1/domains/{DOMAIN}/records/A"
    response = requests.put(url, json=dns_data, headers=headers)
    print(response.status_code, response.text)

if __name__ == "__main__":
    # Get Load Balancer DNS name from AWS (replace with actual command)
    load_balancer_dns = "your-load-balancer-dns-name.amazonaws.com" 
    update_dns(load_balancer_dns)

