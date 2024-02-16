import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests
import re
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from api import gemApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os


def video_links(url):
    '''Function that returns a list of links in the playlist'''
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())) 
    links_list = []
    driver.get(url)
    # Scroll down to load all the videos in the playlist
    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the page to load

    # Get the HTML source of the page
    html_links = driver.page_source
    soup_links = BeautifulSoup(html_links, 'html.parser')

    # Videos link from the playlist
    for a in soup_links.find_all('a',{'class': 'yt-simple-endpoint inline-block style-scope ytd-thumbnail'} ,href=True):
        links_list.append('https://www.youtube.com'+ a['href'])

    # Close the browser window
    driver.quit()
    return links_list

playlist_url = input('Enter the url of the playlist: ')
links = video_links(playlist_url)

#########################################################################################
################################## Vid id and transcript ###############################################
#########################################################################################

def video_id(url):
    # Regular expression to match YouTube video ID in various URL formats
    pattern = (
        r"(?:https?://)?(?:www\.)?(?:youtube\.com/.*[?&]v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/watch\?v=|youtube\.com/e/|youtube\.com/user/.*/u/\d+/|y2u\.be/|youtube\.com/user/.*/U/\w{11})?([a-zA-Z0-9_-]{11})"
    )
    match = re.search(pattern, url)
    return str(match.group(1))


# Get the transcript using the extracted video ID
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = ' '.join([entry['text'] for entry in transcript])
    return text

# Finding the video title:
def vid_title(url): 
      
    # getting the request from url 
    r = requests.get(url) 
      
    # converting the text 
    s = BeautifulSoup(r.text, "html.parser") 
      
    # finding meta info for title 
    title = s.find("title").text

    return title
#########################################################################################
################################## ai integration and final notes###############################################
#########################################################################################

def ai_notes(vid_transcript):
    os.environ['GOOGLE_API_KEY'] = gemApi
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=64000, chunk_overlap=1000)
    texts = text_splitter.create_documents([vid_transcript])

    chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=False)
    chain.llm_chain.prompt.template = \
    """Could you please provide a summary of the given text, including all key points and supporting details? 
    The summary should be comprehensive and accurately reflect the main message and arguments presented in the original text, 
    there is no restriction on the length of the text as long as it contains all the necessary information and dialogues:
    "{text}"

    """
    notes = chain.run(texts)
    return notes

#########################################################################################
##################################final notes###############################################
#########################################################################################


for link in links:
    video_title = vid_title(link)
    video_id_result = video_id(link)
    transcript = get_transcript(video_id_result)
    final_notes = ai_notes(transcript)
    file_name = re.sub(r'[\/:*?"<>|]', '_', video_title) + ".txt"
    with open (file_name, 'w') as file:
        file.write(final_notes)

#####################################################################################################
####################################### TO-DOs ######################################################
#####################################################################################################
# host on kaggle.