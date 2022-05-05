"""
Scrapes ascii art from https://www.asciiart.eu/
Puts it all into data/ dir in subfolders based on category
"""

from bs4 import BeautifulSoup
import requests
import urllib.parse
import os


baseurl = "https://www.asciiart.eu/"
mainpage = requests.get("https://www.asciiart.eu/")
soup = BeautifulSoup(mainpage.content, features='html.parser')

output_dir = "./data/"


def get_and_save_ascii(soup: BeautifulSoup, href):
    art = soup.find("div", attrs={"class":"asciiarts mt-3"})
    arts = art.findAll("pre")
    for i, pic in enumerate(arts):
        pic_str: str = str(pic.contents[0])

        #print(pic_str)

        # Saves to disk
        path = os.path.join(output_dir, href.removeprefix("/"))
        filename = os.path.join(path, "{}.txt".format(i))
        print("Saving pic to {}".format(filename))
        os.makedirs(path, exist_ok=True)
        with open(filename, "w") as f:
            f.write(pic_str)
        print("Saved {}".format(filename))


def recurse_categories(soup: BeautifulSoup, href):
    s = Subcategories(soup)
    if s.no_cats:
        # Gets ascii
        get_and_save_ascii(soup, href)
    for subcat in iter(s):
        url = urllib.parse.urljoin(baseurl, subcat)
        print("Crawling {}".format(url))
        page = requests.get(url)
        soup = BeautifulSoup(page.content, features='html.parser')
        recurse_categories(soup, subcat)


class Subcategories:
    """Iterate subcategory links"""
    def __init__(self, main_page_soup):
        self.links = main_page_soup.find(
            "div", attrs={"class": "directory-columns"}
        )
        if self.links is None:
            self.no_cats = True
        else:
            self.no_cats = False
            self.links = self.links.find_all("li")

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.no_cats:
            raise StopIteration
        if self.i >= len(self.links):
            raise StopIteration
        href = self.links[self.i].find_all(href=True)
        if len(href) > 1 or len(href) == 0:
            raise Exception("incorrect number of links in category")
        href = href[0].get('href')
        if href.startswith("https://"):
            raise Exception("Href is an incorrect format!{}".format(href))
        self.i += 1
        return href


recurse_categories(soup, None)
