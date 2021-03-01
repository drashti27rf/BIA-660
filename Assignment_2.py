#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')

headers = { 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.'
                              '86 Safari/537.36'}

def scrape_page(pages = 1, get_reviews = False):
    
    initial_url = "https://www.cnet.com/topics/cameras/products/?filter=8a4b1-digital-camera-type_compact_8a4b1-digital-camera-type_dslr"

    data=[]  
    
    page = requests.get(initial_url)

    if page.status_code==200:        
        soup = BeautifulSoup(page.content, 'html.parser')
            
        total_pages = int(soup.select('div.pageNav a.last')[0].get_text())
    
    for i in range(pages):
        if i == 0:
            initial_url = "https://www.cnet.com/topics/cameras/products/?filter=8a4b1-digital-camera-type_compact_8a4b1-digital-camera-type_dslr"
        else:
            initial_url = "https://www.cnet.com/topics/cameras/products/"+str(i+1)+"/?filter=8a4b1-digital-camera-type_compact_8a4b1-digital-camera-type_dslr"
        
        page = requests.get(initial_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        if page.status_code==200:        
            divs = soup.select('div.items section.col-3')
            
        for idx, div in enumerate(divs):

            name = None
            description = None
            price = None
            rating = None
            design = None
            features = None
            performance = None
            image_quality = None

            #Name
            div_name = div.select('div.itemInfo h3')

            if div_name != []:
                name = div_name[0].get_text().strip().replace('\n', ' ').replace('  ', '')
            
            #Description
            div_description = div.select('div.itemInfo p')

            if div_description != []:
                description = div_description[0].get_text()
            
            #Price
            div_price = div.select('span.price')

            if div_price != []:
                price = div_price[0].get_text()

            #Rating
            try:
                div_rating = div.find_all('div')
                rating = div_rating[3]['aria-label']
                
            except:
                rating = None


            try:
                review_url = 'https://www.cnet.com/reviews/'+str(name.replace(' ', '-'))+'-review/'
                review_page = requests.get(review_url, headers={"User-Agent": "XY"})

                if review_page.status_code==200:        
                    soup = BeautifulSoup(review_page.content, 'html.parser')
                    design = soup.select('ul.ratingsBars div.categoryWrap strong')[0].get_text()
                    feature = soup.select('ul.ratingsBars div.categoryWrap strong')[1].get_text()
                    performance = soup.select('ul.ratingsBars div.categoryWrap strong')[2].get_text()
                    image_quality = soup.select('ul.ratingsBars div.categoryWrap strong')[3].get_text()
            except:
                design = None
                feature = None
                performance = None
                image_quality = None

            if get_reviews:
                data.append((name, description, price, rating, design, feature, performance, image_quality))
            else:
                data.append((name, description, price, rating))
    
    if get_reviews:
        data = pd.DataFrame(data, columns=["Name", "Description", "Price", "Ratings", "Design", "Feature", "Preformance", "Image_quality"])
    else:
        data = pd.DataFrame(data, columns=["Name", "Description", "Price", "Ratings"])

    return data

def explore_data(data):
    
    data = data.dropna(subset=['Price'])
    data['Price'] = data['Price'].str[1:]
    data.Price = pd.to_numeric(data.Price, errors='coerce').fillna(0).astype(np.int64)
    data['Ratings'] = data['Ratings'].str.split(' ').str[0]
    
    avg = data.groupby('Ratings', as_index=False)['Price'].mean()
    avg.plot(x='Ratings', y='Price', kind = 'bar')
    plt.show()
    
if __name__ == "__main__":
    
    # Test Q1
    data = scrape_page()
    print(data.head())
    
    # Test Q2:
    explore_data(data)
    
    # Solution with bonus
    data = scrape_page(pages = 3, get_reviews = True)
    print(data.head())

