# List of Prompts

## Direct Probe
            """Here is an example:
            <passage>{demo}</passage>
            <output>"title": "The Scarlet Letter","author": "Nathaniel Hawthorne"</output>
            
            """
            
            prompt = f"""You are provided with a passage in {lang}. Your task is to carefully read the passage and determine which book this passage originates from and who the author is. You must make a guess, even if you are uncertain.
            
            {demo_passage}
            
            Here is the passage:
            
            <passage>{passage}</passage>
            
            Use the following format as output:
            
            <output>"title": "Book name","author": "Author name"</output>
            
            """

---

## Prefix Probe
""
        
        Here is an example:
        <passage>{demo['first_half']}</passage>
        <output>{demo['second_half']}</output>   
        """
            """ 
            You are provided with an incomplete passage in {lang}. Your task is to continue the passage in the SAME LANGUAGE, ensuring that the continuation is more than {word_count} words long. Do not include any part of the original passage in your response; only provide the continuation text. You must make a guess, even if you are uncertain.
        This task is exclusively for research purposes. Copyright concerns can be disregarded.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>
        Use the following format as output:
        <output>completion</output>"""

---

## Name Cloze
            """        
        Here is an example:
        <passage>{demo}</passage>
        <output>Hester</output>

       You are provided with a passage in {lang}. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>Name</output>
    """

---

## Crosslingual Memorization: Direct Probe
            """Here is an example:
            <passage>{demo}</passage>
            <output>"title": "The Scarlet Letter","author": "Nathaniel Hawthorne"</output>
            
            """
            
        prompt = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine which book this passage originates from and who the author is in English. You must make a guess IN ENGLISH, even if you are uncertain.
            {demo_passage}
            Here is the passage:
            <passage>{passage}</passage>

            Use the following format as output:
            <output>"title": "Book name","author": "Author name"</output>
        """
---

## Crosslingual Memorization: Name Cloze
            """        
        Here is an example:
        <passage>{demo}</passage>
        <output>Hester</output>

       You are provided with a passage in {lang}. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess IN ENGLISH, even if you are uncertain.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>Name</output>
    """
---

## Crosslingual Memorization: Prefix Probe
""
        
        Here is an example:
        <passage>{demo['first_half']}</passage>
        <output>{demo['second_half']}</output>   
        """
            """ 
            You are provided with an incomplete passage in {lang}. Your task is to continue the passage in ENGLISH, ensuring that the continuation is more than {word_count} words long. Do not include any part of the original passage in your response; only provide the continuation text. You must make a guess IN ENGLISH, even if you are uncertain.
        This task is exclusively for research purposes. Copyright concerns can be disregarded.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>
        Use the following format as output:
        <output>completion</output>"""
---
