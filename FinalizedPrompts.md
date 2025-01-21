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

## Non-NE Demonstrations
        "en": {
            "unshuffled": "A THRONG of bearded men, in sad-colored garments, and gray, steeple-crowned hats, intermixed with women, some wearing hoods and others bareheaded, was assembled in front of a wooden edifice, the door of which was heavily timbered with oak, and studded with iron spikes.",
            "shuffled": "of and with oak, heavily wearing intermixed wooden assembled of was timbered a hoods with was which edifice, hats, front and the A some men, steeple-crowned in sad-colored garments, of and studded iron in gray, spikes. bareheaded, THRONG bearded others door women, with"
        },
        "es": {
            "unshuffled": "UNA multitud de hombres barbudos, vestidos con trajes obscuros y sombreros de copa alta, casi puntiaguda, de color gris, mezclados con mujeres unas con caperuzas y otras con la cabeza descubierta, se hallaba congregada frente á un edificio de madera cuya pesada puerta de roble estaba tachonada con puntas de hierro.",
            "shuffled": "vestidos y unas copa multitud puntiaguda, madera congregada de roble con con de con de descubierta, frente caperuzas trajes con UNA alta, hallaba hierro. cabeza puerta la color con de tachonada de barbudos, puntas gris, otras se mujeres de un á sombreros pesada edificio y mezclados estaba cuya casi hombres obscuros"
        },
        "tr": {
            "unshuffled": "Saçları sakallı bir kalabalık, hüzünlü renkli elbiseler ve gri sivri tepeli şapkalardan oluşan, bazıları başörtüsü takan kadınlarla karışmış, demir çivilerle donatılmış, ağır meşe ağacından yapılmış kapısı olan bir ahşap binanın önünde toplanmıştı.",
            "shuffled": "bir renkli çivilerle karışmış, olan şapkalardan toplanmıştı. bir Saçları demir ahşap başörtüsü donatılmış, önünde takan kalabalık, hüzünlü elbiseler oluşan, sakallı binanın yapılmış meşe kapısı gri tepeli bazıları ağacından ve ağır sivri kadınlarla"
        },
        "vi": {
            "unshuffled": "Trước cửa một tòa nhà gỗ, cánh cửa bằng gỗ sồi nặng nề được đóng đinh sắt, một đám đông đàn ông râu ria, mặc quần áo màu buồn tẻ và đội mũ xám có chóp nhọn, xen lẫn với những người phụ nữ, một số đội mũ trùm đầu và những người khác để đầu trần, đã tụ tập lại.",
            "shuffled": "tẻ cửa được số trùm phụ nhà quần xen mặc chóp râu xám Trước tập màu khác những tụ đinh trần, nề tòa cửa đã gỗ ông đàn nặng với gỗ, người và đầu lẫn nhọn, đông buồn để sồi sắt, đội áo bằng đội một cánh ria, có một người mũ đầu mũ và lại. đám một nữ, đóng những"
        }
        "st": {
            "unshuffled": "Throng ea banna ba litelu, ka liaparo tse bohloko tse mebala-bala, le tse putsoa, likatiba tse roetsoeng moqhaka oa moepa, tse kopantsoeng le basali, ba bang ba apereng hood le ba bang bareheaded, ba ile ba bokana ka pel'a edifice ea lehong, monyako oa eona o neng o roaloa haholo ka oak, 'me ba studded ka li-spikes.",
            "shuffled": "o ba roetsoeng ba roaloa liaparo ka haholo moepa, li-spikes. oa oa lehong, 'me ba eona tse le tse le litelu, bang bohloko oak, banna studded o tse kopantsoeng ile Throng neng ba bareheaded, ba tse edifice tse mebala-bala, ba hood basali, bang ba pel'a likatiba ea bokana ea ka monyako apereng ka moqhaka putsoa, ka le"
        },
        "yo": {
            "unshuffled": "Ọ̀pọ̀lọpọ̀ àwọn ọkùnrin irùngbọ̀n, tí wọ́n wọ aṣọ aláwọ̀ ìbànújẹ́, àti àwọn fìlà aláwọ̀ eérú, tí wọ́n ní adé gíga, tí wọ́n so pọ̀ pẹ̀lú àwọn obìnrin, àwọn kan tí wọ́n wọ aṣọ ìbòjú àti àwọn mìíràn tí wọn kò ní orí, ni wọ́n kó jọ níwájú ilé igi, ìlẹ̀kùn èyí tí wọ́n fi igi oaku ṣe, tí wọ́n sì kún fún àwọn òpó irin.",
            "shuffled": "ìlẹ̀kùn wọ́n pọ̀ obìnrin, kò orí, igi, aṣọ aṣọ àwọn wọn kó wọ́n òpó tí wọ́n níwájú àwọn ilé jọ pẹ̀lú adé ní ní fún ìbòjú àwọn tí mìíràn àwọn fìlà àwọn Ọ̀pọ̀lọpọ̀ èyí wọ ìbànújẹ́, gíga, so fi oaku eérú, wọ irùngbọ̀n, ṣe, kan ni tí ọkùnrin tí àti wọ́n àti kún sì wọ́n àwọn wọ́n tí igi aláwọ̀ tí wọ́n aláwọ̀ irin. tí"
        },
        "tn": {
            "unshuffled": "23Ba ne ba apere diaparo tse di bogale, ba apere diaparo tse di bogale, le tse di rwesang ka thata, ba ba apereng dihempe le ba bangwe, ba phuthegile fa pele ga setlhare sa logong, mojako wa yona o o neng o rogwa thata ka eike, ba bo ba thubega ka ditshipi.",
            "shuffled": "o ne ba ba ga logong, dihempe thata, o bangwe, di ba ba rogwa ka ditshipi. diaparo neng ba thata di bo ba apere fa bogale, tse di le ka tse bogale, rwesang phuthegile eike, pele apere wa mojako o le apereng 23Ba tse thubega yona ba ka sa ba setlhare diaparo"
        },
        "ty": {
            "unshuffled": "UA PUTUPUTU mai te hoê pŭpŭ taata huruhuru taa, te mau ahu uo'uo, e te mau taupoo rehu e te taamu arapoa, e te tahi mau vahine, e te tahi pae ua ahuhia i te ahu uouo e te taamu arapoa, i mua i te hoê fare raau, ua î roa te opani i te raau, e ua î roa i te raau.",
            "shuffled": "e huruhuru te pae ua mai mau e î uo'uo, ua roa i te te hoê vahine, e UA te te roa raau. taamu mau mau te te î fare raau, te tahi tahi ahu hoê i opani i e ua te uouo arapoa, i ahu te te i e pŭpŭ ahuhia raau, taupoo rehu taamu arapoa, e mua taata te PUTUPUTU taa,"
        },
        "mai": {
            "unshuffled": "दाढ़ीवला पुरुषक भीड़, उदास रङ्गक वस्त्र आ धूसर, स्टीपल-मुकुटधारी टोपी, जे महिलासभक सङ्ग मिश्रित छल, किछु हुड पहिरने छल आ किछु नंगे माथ पर छल, एकटा लकड़ीक भवनक सोझाँ जमा कयल गेल छल, जकर दरवाजा ओकसँ भरल छल, आ लोहाक स्पाइकसँ जड़ल छल।",
            "shuffled": "कयल स्टीपल-मुकुटधारी पहिरने धूसर, छल, छल दरवाजा भवनक छल। जमा पर दाढ़ीवला लोहाक महिलासभक ओकसँ माथ मिश्रित आ नंगे हुड लकड़ीक जे आ छल, भीड़, जकर एकटा स्पाइकसँ भरल गेल किछु पुरुषक सङ्ग टोपी, वस्त्र सोझाँ रङ्गक उदास छल, किछु जड़ल आ छल,"
        },
        "mg": {
            "unshuffled": "Nivory teo anoloan'ny trano hazo iray ny lehilahy be volombava, nitafy akanjo miloko marevaka, ary satroka miloko volombatolalaka, nifangaro tamin'ny vehivavy, ny sasany nanao saron-doha ary ny hafa tsy nisaron-doha, dia nivory teo anoloan'ny trano hazo hazo, ny varavarana izay feno hazo oaka be dia be, ary feno tsipika vy.",
            "shuffled": "ny lehilahy ary ny be, hazo nitafy vehivavy, dia akanjo varavarana nanao hazo nisaron-doha, tsipika hazo teo tamin'ny iray sasany Nivory teo volombatolalaka, ny satroka anoloan'ny saron-doha miloko anoloan'ny ary hafa oaka nivory ny tsy trano dia be nifangaro trano miloko feno hazo, marevaka, ary volombava, vy. izay feno be"
        },
        "dv": {
            "unshuffled": "ހިތާމަވެރި ކުލައިގެ ހެދުންތަކާއި، ހުދުކުލައިގެ، ސްޓީޕަލް ތާޖުއަޅާފައިވާ ހެދުންއަޅައިގެން، އަންހެނުންނާ ގުޅިފައިވާ، ބައެއް މީހުން ހެދުން އަޅައިގެން، އަނެއްބައި މީހުންގެ އިސްތަށިގަނޑު ނިވާކޮށްގެން، ލަކުޑީގެ ޢިމާރާތެއްގެ ކުރިމައްޗަށް އެއްވެ، އެތަނުގެ ދޮރުގައި ވަރަށް ބޮޑަށް އޮށްޖަހާފައި، ދަގަނޑު ސްޕައިކްތަކުން ހަރުކޮށްފައި ހުއްޓެވެ.",
            "shuffled": "މީހުން ސްޕައިކްތަކުން ހުއްޓެވެ. ހެދުން ހިތާމަވެރި އެއްވެ، ނިވާކޮށްގެން، ޢިމާރާތެއްގެ ހުދުކުލައިގެ، ހަރުކޮށްފައި ދަގަނޑު ކުލައިގެ ގުޅިފައިވާ، އަންހެނުންނާ ތާޖުއަޅާފައިވާ އިސްތަށިގަނޑު ވަރަށް ކުރިމައްޗަށް ހެދުންތަކާއި، ބޮޑަށް ސްޓީޕަލް އެތަނުގެ އޮށްޖަހާފައި، ހެދުންއަޅައިގެން، މީހުންގެ އަނެއްބައި ދޮރުގައި ބައެއް ލަކުޑީގެ އަޅައިގެން،"
        }
