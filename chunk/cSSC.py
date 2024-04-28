# ! pip install -q -U google-generativeai
# -*- coding: utf-8 -*-
import google.generativeai as genai
GOOGLE_API_KEY = 'YOUR API KEY'

genai.configure(api_key=GOOGLE_API_KEY)

import tqdm
import pathlib
import textwrap
from IPython.display import display
from IPython.display import Markdown
import pandas as pd
import numpy as np
import os, pickle, json, time

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def make_prompt(context):
  input_prompt = f'''
  This is a sequential classification task for medical abstracts. There are only 5 labels, namely: OBJECTIVE, BACKGROUND, METHODS, RESULTS, CONCLUSIONS. I want to classify each sentence with its corresponding label. Note that each sentence corresponds to ONE AND ONLY ONE LABEL.

  To take for example, below are 10 abstracts and their sentences, with the format of LABEL, TEXT.

  ABSTRACT 1
  OBJECTIVE	abcxyz
  METHODS	abcxyz
  RESULTS	abcxyz
  ...
  
  Here is the abstract and its sentences seperated by ' '. Return the correct label for the sentences!
  {context}

  (Note that there are only 5 labels, namely: OBJECTIVE, BACKGROUND, METHODS, RESULTS, CONCLUSIONS. Return ONLY ONE label for each sentence and nothing else. Return the Number of the sentence and its Label. E.g., 1 - Methods).
  '''

  return input_prompt

# Rest of your code goes here

def create_model(temperature: float = 0.0, max_output_tokens: int = 2048, 
                 safety_settings: str = None, model_name: str = "gemini-pro"):
  generation_config = {
    "temperature": temperature,
    "max_output_tokens": max_output_tokens, # 2048
  }
  
  if safety_settings == None:
      safety_settings = [
        {
          "category": "HARM_CATEGORY_HARASSMENT",
          "threshold": "block_none" # "BLOCK_MEDIUM_AND_ABOVE" 
        },
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "threshold": "block_none" # "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "threshold": "block_none" # "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
          "threshold": "block_none" # "BLOCK_MEDIUM_AND_ABOVE"
        },
      ]

  model = genai.GenerativeModel(model_name,
                                generation_config=generation_config,
                                safety_settings=safety_settings)
  return model

def generate_content_with_df(input_df: pd.DataFrame,
                             model, temperature: float = 0.0):

  output_csv_file = 'output.csv'

  output_df = {'text':[], 'prediction':[]}

  # Loop through the df row [text]
  for index, row in tqdm.tqdm(input_df.iterrows(), total = len(input_df)):
      # append the [text] to the prompt
      prompt = make_prompt(row['Text'])

      try:
      # generate the response
          response = model.generate_content(prompt).text
      except:
          response = np.nan
      
      # append the response.text to the prediction collumn
      # output_df = output_df.append({'text': row['text'],
                                    # 'prediction': response},ignore_index=True)
      output_df['text'].append(row['Text'])

      if isinstance(response, float):
        response = np.nan

      elif isinstance(response, str):
      # Check if response is an empty string after stripping
        if response.strip() == '':
          response = np.nan

      output_df['prediction'].append(response)

  # Save output DataFrame to CSV file
  output_df = pd.DataFrame(output_df)
  output_df.to_csv(output_csv_file, index=False)

  return output_csv_file

# input
input_file = 'test_0000_10.csv'
input_df = pd.read_csv(input_file)

# model
model = create_model()

# output
response_output_file = generate_content_with_df(input_df, model)
