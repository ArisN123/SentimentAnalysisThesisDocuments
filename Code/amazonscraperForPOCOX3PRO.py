import requests
from bs4 import BeautifulSoup
import csv

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
}

url = 'https://www.amazon.nl/Xiaomi-Poco-X3-Pro-Smartphone/product-reviews/B08YJFSHFM'
##update URL based on amazon product you are looking to scrape

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

with open('reviews.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Review Title', 'Review Text', 'Review Rating'])

    reviews = soup.find_all('div', {'data-hook': 'review'})

    for review in reviews:
        title = review.find('a', {'data-hook': 'review-title'}).text.strip()
        text = review.find('span', {'data-hook': 'review-body'}).text.strip()
        rating = review.find('i', {'data-hook': 'review-star-rating'}).text.strip()

        writer.writerow([title, text, rating])
