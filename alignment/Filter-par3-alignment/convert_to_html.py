import pandas as pd
import os

def csv_to_html(csv_file, output_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # HTML head and style
    html_content = """
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<style type="text/css">
</style>
</head>
<body>
    """

    # Convert the DataFrame to an HTML table
    html_table = df.to_html(index=False, classes='dataframe')

    # Close the HTML structure
    html_content += html_table + "</body></html>"
    
    # Write the generated HTML to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def get_file_names(directory):
    txt_file_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith('.csv'):
            txt_file_names.append(item.replace('_aligned.csv', '').replace('_para', ''))
    return txt_file_names

            
if __name__ == '__main__':
    alignment_lvl= 'sent' # para / sent
    titles = get_file_names(f'/Filter-par3-alignment/aligned_filtered/{alignment_lvl}')
    print(titles)
    for title in titles:
        print(f'processing {title}')
        csv_to_html(f'/Filter-par3-alignment/ner/{title}/{title}_ner.csv', 
                    f'/Filter-par3-alignment/html/{alignment_lvl}/{title}_aligned.html')

