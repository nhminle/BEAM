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

## Crosslingual Memorization: Unmasked Demonstrations
            "st": {
            "unshuffled": "Me re tlameha ho re, seared bosom ea Hester haholo, hoo mohlomong ho ne ho e-na le' nete ho feta menyenyetsi ho feta ho se lumele ha rona ea kajeno e ka ba tšekamelo ea ho lumela.",
            "shuffled": "se ho Me ea re ea ho kajeno hoo ea haholo, ho rona feta ho ho ho ne tšekamelo e lumela. feta ha seared e-na ka nete le' bosom re, mohlomong ho Hester tlameha ba lumele menyenyetsi"
        },
        "yo": {
            "unshuffled": "Àti pé a gbọ́dọ̀ nílò láti sọ pé, ó mú àyà Hester jinlẹ̀, pé bóyá òtítọ́ púpọ̀ wà nínú àròsọ ju àìgbàgbọ́ ìgbàlódé wa lọ lè fẹ́ láti gbà.",
            "shuffled": "pé láti wà àyà ìgbàlódé nílò púpọ̀ mú wa pé Àti pé, a lọ àròsọ ó gbà. láti fẹ́ Hester gbọ́dọ̀ òtítọ́ ju nínú jinlẹ̀, sọ bóyá lè àìgbàgbọ́"
        },
        "tn": {
            "unshuffled": "Mme re tshwanetse go re, re ne ra re, go ne go le thata gore re nne le tumelo ya ga Jehofa e e neng e le mo go yone, e ka tswa e le boammaaruri jo bogolo go feta tumelo ya rona ya gompieno.",
            "shuffled": "rona tumelo e yone, re e le e re le ga go gore go ne le mo tshwanetse ka boammaaruri e Mme nne re Jehofa ya gompieno. go jo e re, thata ne tswa ra bogolo re, neng go le feta ya ya go tumelo"
        },
        "ty": {
            "unshuffled": "E e tia ia tatou ia parau e, ua mauiui roa te ouma o Hester, e peneia'e ua rahi a'e te parau mau i roto i te parau i faahitihia i to tatou tiaturi ore no teie nei tau.",
            "shuffled": "roto mau i e, e tia teie ouma to tau. te o e parau faahitihia rahi i tatou ua te tiaturi te parau E peneia'e i no a'e tatou ua Hester, ia nei mauiui ia ore roa i parau"
        },
        "mai": {
            "unshuffled": "आ हमरासभकेँ ई कहबाक आवश्यकता अछि जे ई हेस्टरक छातीकेँ एतेक गहराई सँ प्रभावित कयलक, जे शायद अफवाहमे ओहिसँ बेसी सत्य छल जतेक हमर आधुनिक अविश्वास स्वीकार करय लेल इच्छुक भऽ सकैत अछि।",
            "shuffled": "जे हमरासभकेँ भऽ छल आ सकैत इच्छुक आवश्यकता हमर लेल ओहिसँ कयलक, जतेक सँ गहराई बेसी कहबाक करय जे स्वीकार एतेक अविश्वास शायद सत्य अफवाहमे ई अछि ई आधुनिक छातीकेँ प्रभावित अछि। हेस्टरक"
        },
        "mg": {
            "unshuffled": "Ary tsy maintsy mila miteny isika hoe, tena nampivoaka lalina ny tratran'i Hester izany, ka angamba nisy fahamarinana bebe kokoa tao anatin'ilay tsaho noho ny tsy finoana maoderina izay mety ho mora miaiky ny tsy finoana maoderina.",
            "shuffled": "izany, nisy mora tratran'i nampivoaka finoana ka izay mety kokoa isika tsy tao ny finoana maoderina miaiky tsy lalina hoe, ny Ary anatin'ilay ny bebe maoderina. tsy miteny Hester angamba fahamarinana noho mila maintsy ho tena tsaho"
        },
        "dv": {
            "unshuffled": "އެހެންވީމާ، އަޅުގަނޑުމެން ދަންނަވާލަން އެބަޖެހޭ، އެއީ ހެސްޓަރގެ ސިކުނޑި އެހާ ބޮޑަށް ކަނޑުވާލި ކަމެއް، އެއީ، އަޅުގަނޑުމެންގެ މިޒަމާނުގެ ނުތަނަވަސްކަން ޤަބޫލުކުރަން ޝައުޤުވެރިވާ ވަރަށްވުރެ، އެ ވާހަކަތަކުގައި ޙަޤީޤަތެއް އޮވެދާނެ.",
            "shuffled": "ޤަބޫލުކުރަން ކަމެއް، މިޒަމާނުގެ ހެސްޓަރގެ ޝައުޤުވެރިވާ އަޅުގަނޑުމެން އޮވެދާނެ. ނުތަނަވަސްކަން ޙަޤީޤަތެއް އަޅުގަނޑުމެންގެ އެބަޖެހޭ، އެހާ ދަންނަވާލަން ވަރަށްވުރެ، ވާހަކަތަކުގައި ކަނޑުވާލި އެއީ ސިކުނޑި އެހެންވީމާ، އެ ބޮޑަށް އެއީ،"
        }
