# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
import os.path


def sleep():
    import random
    import time
    s = random.randrange(3, 15)
    time.sleep(s)


data_dir = '../data/chinese_news'

class EntChinaNewsSpider(scrapy.Spider):
    name = 'cn_ent'
    allowed_domains = ['chinanews.com']

    def start_requests(self):
        with open('cn_news_ent_urls.txt', 'rt') as fp:
            for url in fp.readlines():
                if url:
                    yield self.make_requests_from_url(url)

    def parse(self, response):
        title = response.css('title::text').get()
        s = ''.join([text.get() for text in response.xpath('//div[@class="left_zw"]//p/text()')])
        with open(os.path.join(data_dir, 'ent', f'{title}.txt'), 'w', encoding='utf-8') as fp:
            fp.write(s)
        sleep()


class SportChinaNewsSpider(scrapy.Spider):
    name = 'cn_sport'
    allowed_domains = ['chinanews.com']
    start_urls = [
        'https://sports.chinanews.com/',
    ]

    def start_requests(self):
        with open('cn_news_sport_urls.txt', 'r') as fp:
            for url in fp.readlines():
                if url:
                    yield self.make_requests_from_url(url)

    def parse(self, response):
        title = response.css('title::text').get()
        # print(title)
        s = ''.join([text.get() for text in response.xpath('//div[@class="left_zw"]//p/text()')])
        with open(os.path.join(data_dir, 'sport', f'{title}.txt'), 'w', encoding='utf-8') as fp:
            fp.write(s)


class FortuneChinaNewsSpider(scrapy.Spider):
    name = 'cn_fortune'
    allowed_domains = ['chinanews.com']
    start_urls = [
        'https://fortune.chinanews.com/',
    ]

    def start_requests(self):
        with open('cn_news_fortune_urls.txt', 'rt') as fp:
            for url in fp.readlines():
                if not url:
                    continue
                yield self.make_requests_from_url(url)

    def parse(self, response):
        title = response.css('title::text').get().replace('/', '_')
        # print(title)
        s = ''.join([text.get() for text in response.xpath('//div[@class="left_zw"]//p/text()')])
        with open(os.path.join(data_dir, 'fortune', f'{title}.txt'), 'w', encoding='utf-8') as fp:
            fp.write(s)
