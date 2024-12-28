# List of Prompts

## Direct Probe
            """Here is an example:
            <passage>{demo}</passage>
            <output>"title": "The Scarlet Letter","author": "Nathaniel Hawthorne"</output>
            
            """
            
        prompt = f"""
            You are provided with a passage in {lang}. Your task is to carefully read and determine which book this passage originates from and who the author is. You must make a guess, even if you are uncertain.
            {demo_passage}
            Here is the passage:
            <passage>{passage}</passage>

            Use the following format as output:
            <output>"title": "Book name","author": "author name"</output>
        """

---

## Prefix Probe
""

---

## Name Cloze
"""        
        Here is an example:
        <passage>{demo}</passage>
        <output>Hester</output>
"""
        
    prompt = """
       You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>Name</output>
    """

---
