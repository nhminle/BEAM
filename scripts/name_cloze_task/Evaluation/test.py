import pandas as pd

# List of titles and their release dates (manually curated based on known publication years)
titles_and_dates = [
    {"Title": "Sense_and_sensibility", "Release Date": "1811-01-01"},
    {"Title": "Adventures_of_Sherlock_Holmes", "Release Date": "1892-10-14"},
    {"Title": "Adventures_of_Huckleberry_Finn", "Release Date": "1884-12-10"},
    {"Title": "The_Great_Gatsby", "Release Date": "1925-04-10"},
    {"Title": "Frankenstein", "Release Date": "1818-01-01"},
    {"Title": "Paper_Towns", "Release Date": "2008-10-16"},
    {"Title": "Dune", "Release Date": "1965-08-01"},
    {"Title": "The_Picture_of_Dorian_Gray", "Release Date": "1890-06-20"},
    {"Title": "The_Handmaid_s_Tale", "Release Date": "1985-09-01"},
    {"Title": "Alice_in_Wonderland", "Release Date": "1865-11-26"},
    {"Title": "Pride_and_Prejudice", "Release Date": "1813-01-28"},
    {"Title": "1984", "Release Date": "1949-06-08"},
    {"Title": "The_Boy_in_the_Striped_Pyjamas", "Release Date": "2006-01-05"},
    {"Title": "Percy_Jackson_The_Lightning_Thief", "Release Date": "2005-06-28"},
    {"Title": "Dracula", "Release Date": "1897-05-26"},
    {"Title": "Of_Mice_and_Men", "Release Date": "1937-01-01"},
    {"Title": "A_Tale_of_Two_Cities", "Release Date": "1859-04-30"},
    {"Title": "A_thousand_splendid_suns", "Release Date": "2007-05-22"},
    {"Title": "Harry_Potter_and_the_Deathly_Hallows", "Release Date": "2007-07-21"},
    {"Title": "Fahrenheit_451", "Release Date": "1953-10-19"},\
]

# Convert to DataFrame
titles_df = pd.DataFrame(titles_and_dates)

# Save to CSV
output_path = "/Users/minhle/Umass/ersp/Evaluation/eval/csv/release_date.csv"
titles_df.to_csv(output_path, index=False)

output_path
