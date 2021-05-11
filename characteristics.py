from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PATH = r'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(PATH)

driver.get("https://www.hillspet.co.uk/dog-care/breeds")
search = driver.find_element_by_id('search_01600624965')
search.send_keys('saint bernard')
search.send_keys(Keys.RETURN)


try:
    search_results = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "st-pcc-results-container"))
    )
    print(search_results.text)
    # articles = search_results.find_elements_by_id("st-pcc-results-container")
    # for article in articles:
    #     print(articles.text)
    #     header = article.find_element_by_class_name("article-search-item--right-column")
    #     print(header.text)
finally:
    pass


