# NLP_Project
Many people check item/product reviews to decide whether to buy it or not, but some reviews may be too long to read. So in this project we are going to summarize the long reviews that is text summarization (including key word extraction) and implement aspect based sentiment analysis on it.
## MODELS USED
TEXT SUMMARIZATION - BartForConditionalGeneration <br />
KEY WORD EXTRACTION - Using Yake library <br />
ASPECT BASED SENTIMENT ANALYSIS - Stanza <br />

Coming to the files that are there in the repository, we have uploaded separate code files for the above the tasks, and we also uploaded other files in which we and written some demo codes initially, on that basis we built our code for the application.

## Requirements
NLTK - 3.8.1 <br />
stanza - 1.7.0 <br />
transformers - 4.35.2 <br />
yake - 0.4.8 <br />

The above are the requirments of versions of libraries that you should have to run the code
And you need to have a active internet to download some pretrained models that are used in the code.

## Conclusion
This is one of the basic approach of integrating text summarization, key word extraction, aspect based sentiment analysis.This prototype can help many users in saving many resources like time and money as it significantly reduces the time used by users to read long reviews and in most of the cases it helps people take wise decisions on choosing the product.

 ## Future Work
We haven't used any domain specific dataset for our prototype. So in future we will create our own dataset which boosts our prototype performance and we also use the dataset to tune the parameters of our pre-trained models. 
