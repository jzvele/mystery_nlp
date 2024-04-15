import torch
from transformers import BertTokenizer, BertModel

relationship_categories = ["colleague", "stepmother", "stepfather", "stepchild", "stepsibling", "neighbor", "employer", "employee", "acquaintance", "friend", "lover", "father", "mother", "daughter", "son", "sibling", "spouse", "accomplice", "rival", "client", "patient", "co-conspirator", "romantic partner", "fiancé", "fiancée", "sibling-in-law","cousin", "niece", "nephew", "uncle", "aunt", "governess", "ward", "wife", "husband", "collaborator", "manipulator",'former lover', "mother-in-law", "servant", "stepson", "stepdaughter", "ex-fiancé", "assistant", "maid", "secretary", "love interest", "solicitor", "look-alike", "alleged father", "alleged son", "father's former lover's son", "grandfather", "trustee", "pseudonym", "blackmailer", "protector", "companion", "half-sibling", "housekeeper", "business associate", "doctor", "former employer", "former employee", "donor", "advisor", "butler", "gardener", "caretaker"]

filename_mapping = {
    'After the Funeral' : 'atf_relationships',
    'Appointment With Death' : 'awd_relationships',
    'Death On the Nile' : 'dotn_relationships',
    'Five Little Pigs' : 'flp_relationships',
    'Hercule Poirot\'s Christmas' : 'hpc_relationships',
    'The ABC Murders' : 'tabcm_relationships',
    'The Hunter\'s Lodge Case' : 'thlc_relationships',
    'The Murder of Roger Ackroyd' : 'tmra_relationships',
    'The Murder on the Links' : 'tmotl_relationships',
    'The Mysterious Affair at Styles' : 'tmaas_relationships',
    'The Mystery of the Blue Train' : 'tmotbt_relationships',
    'Murder On The Orient Express' : 'motoe_relationships',
    'scene_1' : 'scene_1_relationships',
    'scene_2' : 'scene_2_relationships',
    'scene_3' : 'scene_3_relationships',
    'scene_4' : 'scene_4_relationships',
    'scene_5' : 'scene_5_relationships',
    'scene_6' : 'scene_6_relationships',
    'scene_7' : 'scene_7_relationships',
    'scene_8' : 'scene_8_relationships',
    'scene_9' : 'scene_9_relationships',
    'scene_10' : 'scene_10_relationships',
    'scene_11' : 'scene_11_relationships',
    'scene_12' : 'scene_12_relationships'
}

bidirectional_pairs = {
    'thlc_relationships' : 
[
    {
        "PersonA": "Roger Havering",
        "PersonB": "Zoe Havering",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Roger Havering",
        "PersonB": "Mr. Harrington Pace",
        "RelationshipAtoB": "uncle",
        "RelationshipBtoA": "nephew",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Zoe Havering",
        "PersonB": "Mrs. Middleton",
        "RelationshipAtoB": "employee",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Zoe Havering",
        "PersonB": "Mr. Harrington Pace",
        "RelationshipAtoB": "uncle",
        "RelationshipBtoA": "niece",
        "CorrectChoice": "AtoB"
    }
],
'hpc_relationships' : 
[
    {
        "PersonA": "Mrs. Otterbourne",
        "PersonB": "Rosalie Otterbourne",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "mother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Miss Van Schuyler",
        "PersonB": "Cornelia Robson",
        "RelationshipAtoB": "employee",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Louise Bourget",
        "PersonB": "Linnet Doyle",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "maid",
        "CorrectChoice": "BtoA"
    }
],
'tmotl_relationships' : 
[
    {
        "PersonA": "Mr. Renauld",
        "PersonB": "Madame Renauld",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mr. Renauld",
        "PersonB": "Jack Renauld",
        "RelationshipAtoB": "son",
        "RelationshipBtoA": "father",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Jack Renauld",
        "PersonB": "Madame Renauld",
        "RelationshipAtoB": "mother",
        "RelationshipBtoA": "son",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Marthe Daubreuil",
        "PersonB": "Madame Daubreuil",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "mother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Marthe Daubreuil",
        "PersonB": "Madame Beroldy",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "mother",
        "CorrectChoice": "AtoB"
    }
],
'motoe_relationships' : 
[
    {
        "PersonA": "Princess Dragomiroff",
        "PersonB": "Hildegarde Schmidt",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "employee",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Countess Andrenyi",
        "PersonB": "Count Andrenyi",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    }
],
'awd_relationships' : 
[
    {
        "PersonA": "Nadine",
        "PersonB": "Lennox Boynton",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Ginevra Boynton",
        "PersonB": "Mrs. Boynton",
        "RelationshipAtoB": "stepmother",
        "RelationshipBtoA": "stepdaughter",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Boynton",
        "PersonB": "Lennox Boynton",
        "RelationshipAtoB": "stepson",
        "RelationshipBtoA": "stepmother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Boynton",
        "PersonB": "Raymond Boynton",
        "RelationshipAtoB": "stepson",
        "RelationshipBtoA": "stepmother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Boynton",
        "PersonB": "Carol Boynton",
        "RelationshipAtoB": "stepdaughter",
        "RelationshipBtoA": "stepmother",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Lennox Boynton",
        "PersonB": "Carol Boynton",
        "RelationshipAtoB": "stepsister",
        "RelationshipBtoA": "stepbrother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Raymond Boynton",
        "PersonB": "Carol Boynton",
        "RelationshipAtoB": "stepsister",
        "RelationshipBtoA": "stepbrother",
        "CorrectChoice": "AtoB"
    }
],
'tmra_relationships' : 
[
    {
        "PersonA": "Ralph Paton",
        "PersonB": "Mr. Ackroyd",
        "RelationshipAtoB": "stepson",
        "RelationshipBtoA": "stepfather",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mr. Ackroyd",
        "PersonB": "Miss Ackroyd",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "father",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mr. Ackroyd",
        "PersonB": "Ursula Bourne",
        "RelationshipAtoB": "employee",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "BtoA"
    }
],
'tabcm_relationships' : 
[
    {
        "PersonA": "Betty Barnard",
        "PersonB": "Mary Drower",
        "RelationshipAtoB": "niece",
        "RelationshipBtoA": "aunt",
        "CorrectChoice": "BtoA"
    }
],
'atf_relationships' : 
[
    {
        "PersonA": "Timothy Abernethie",
        "PersonB": "Mrs Timothy Abernethie",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs Rosamund Shane",
        "PersonB": "Michael Shane",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    }
],
'flp_relationships' : 
[
    {
        "PersonA": "Caroline Crale",
        "PersonB": "Amyas Crale",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Caroline Crale",
        "PersonB": "Miss Williams",
        "RelationshipAtoB": "governess",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'tmaas_relationships' : 
[
    {
        "PersonA": "Monsieur Lawrence",
        "PersonB": "Mrs. Inglethorp",
        "RelationshipAtoB": "stepmother",
        "RelationshipBtoA": "stepson",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Monsieur Lawrence",
        "PersonB": "Alfred Inglethorp",
        "RelationshipAtoB": "stepfather",
        "RelationshipBtoA": "stepson",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Dorcas",
        "PersonB": "Mrs. Inglethorp",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "servant",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Annie",
        "PersonB": "Mrs. Inglethorp",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "servant",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Alfred Inglethorp",
        "PersonB": "Mrs. Inglethorp",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Cavendish",
        "PersonB": "Mrs. Inglethorp",
        "RelationshipAtoB": "mother-in-law",
        "RelationshipBtoA": "stepdaughter",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Inglethorp",
        "PersonB": "John Cavendish",
        "RelationshipAtoB": "stepson",
        "RelationshipBtoA": "stepmother",
        "CorrectChoice": "BtoA"
    }
],
'scene_9_relationships' : 
[
    {
        "PersonA": "Reginald",
        "PersonB": "Lady Warwick",
        "RelationshipAtoB": "stepmother",
        "RelationshipBtoA": "stepson",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Warwick",
        "PersonB": "William",
        "RelationshipAtoB": "caretaker",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Warwick",
        "PersonB": "Agnes",
        "RelationshipAtoB": "maid",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_8_relationships' : 
[
    {
        "PersonA": "Lady Harrow",
        "PersonB": "Lord Harrow",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Lady Harrow",
        "PersonB": "Jenkins",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Jenkins",
        "PersonB": "Lord Harrow",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "butler",
        "CorrectChoice": "BtoA"
    }
],
'scene_4_relationships' : 
[
    {
        "PersonA": "Mrs. Cadwell",
        "PersonB": "Reginald Stonebrook",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mrs. White",
        "PersonB": "Reginald Stonebrook",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "housekeeper",
        "CorrectChoice": "BtoA"
    }
],
'scene_11_relationships' : 
[
    {
        "PersonA": "Mr. John Collins",
        "PersonB": "Mr. Peter Hancock",
        "RelationshipAtoB": "solicitor",
        "RelationshipBtoA": "client",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Rebecca Collins",
        "PersonB": "Mr. Andrew Collins",
        "RelationshipAtoB": "stepson",
        "RelationshipBtoA": "stepmother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Rebecca Collins",
        "PersonB": "Mr. Peter Hancock",
        "RelationshipAtoB": "solicitor",
        "RelationshipBtoA": "client",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Rebecca Collins",
        "PersonB": "Ms. Martha Stuart",
        "RelationshipAtoB": "housekeeper",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_1_relationships' : 
[
    {
        "PersonA": "Sir Reginald Hugh",
        "PersonB": "Mr. Johnson",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Sir Reginald Hugh",
        "PersonB": "Arthur",
        "RelationshipAtoB": "son",
        "RelationshipBtoA": "father",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Sir Reginald Hugh",
        "PersonB": "Mary",
        "RelationshipAtoB": "maid",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Sir Reginald Hugh",
        "PersonB": "Esther Hugh",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "father",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mr. Johnson",
        "PersonB": "Esther Hugh",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "butler",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Lady Margaret",
        "PersonB": "Arthur",
        "RelationshipAtoB": "nephew",
        "RelationshipBtoA": "aunt",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Lady Margaret",
        "PersonB": "Esther Hugh",
        "RelationshipAtoB": "niece",
        "RelationshipBtoA": "aunt",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mary",
        "PersonB": "Esther Hugh",
        "RelationshipAtoB": "employer",
        "RelationshipBtoA": "maid",
        "CorrectChoice": "BtoA"
    }
],
'scene_5_relationships' : 
[
    {
        "PersonA": "James Glendale",
        "PersonB": "Isabella Glendale",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "father",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "James Glendale",
        "PersonB": "William",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "James Glendale",
        "PersonB": "Dr. Harper",
        "RelationshipAtoB": "doctor",
        "RelationshipBtoA": "patient",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Isabella Glendale",
        "PersonB": "Dr. Harper",
        "RelationshipAtoB": "doctor",
        "RelationshipBtoA": "patient",
        "CorrectChoice": "AtoB"
    }
],
'scene_10_relationships' : 
[
    {
        "PersonA": "Lord Hawkshadow",
        "PersonB": "Mr. Withers",
        "RelationshipAtoB": "advisor",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lord Hawkshadow",
        "PersonB": "Mrs. Gretyl",
        "RelationshipAtoB": "love interest",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Gretyl",
        "PersonB": "Jameson",
        "RelationshipAtoB": "colleague",
        "RelationshipBtoA": "love interest",
        "CorrectChoice": "BtoA"
    }
],
'scene_6_relationships' : 
[
    {
        "PersonA": "Lucille Thornfield",
        "PersonB": "Lady Arabella Maddox",
        "RelationshipAtoB": "aunt",
        "RelationshipBtoA": "niece",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lucille Thornfield",
        "PersonB": "Mr. Wright",
        "RelationshipAtoB": "solicitor",
        "RelationshipBtoA": "client",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lucille Thornfield",
        "PersonB": "Alfred",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lucille Thornfield",
        "PersonB": "Emma",
        "RelationshipAtoB": "housekeeper",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Jasper Thornfield",
        "PersonB": "Lady Arabella Maddox",
        "RelationshipAtoB": "aunt",
        "RelationshipBtoA": "nephew",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Jasper Thornfield",
        "PersonB": "Alfred",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Jasper Thornfield",
        "PersonB": "Emma",
        "RelationshipAtoB": "servant",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Arabella Maddox",
        "PersonB": "Mr. Wright",
        "RelationshipAtoB": "solicitor",
        "RelationshipBtoA": "client",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Arabella Maddox",
        "PersonB": "Alfred",
        "RelationshipAtoB": "servant",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Arabella Maddox",
        "PersonB": "Emma",
        "RelationshipAtoB": "servant",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_3_relationships' : 
[
    {
        "PersonA": "Mrs. Goodwin",
        "PersonB": "Mr. Goodwin",
        "RelationshipAtoB": "husband",
        "RelationshipBtoA": "spouse",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mrs. Goodwin",
        "PersonB": "Miss Grace",
        "RelationshipAtoB": "employee",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mrs. Goodwin",
        "PersonB": "Mister Kent",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_7_relationships' : 
[
    {
        "PersonA": "Lady Margaret",
        "PersonB": "Isabelle",
        "RelationshipAtoB": "daughter",
        "RelationshipBtoA": "mother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Margaret",
        "PersonB": "Victoria",
        "RelationshipAtoB": "maid",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Lady Margaret",
        "PersonB": "Arthur",
        "RelationshipAtoB": "gardener",
        "RelationshipBtoA": "former employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_12_relationships' : 
[
    {
        "PersonA": "Alfred Chesterfield",
        "PersonB": "Marjorie",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Alfred Chesterfield",
        "PersonB": "Edward",
        "RelationshipAtoB": "son",
        "RelationshipBtoA": "father",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Alfred Chesterfield",
        "PersonB": "Eliza",
        "RelationshipAtoB": "governess",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Alfred Chesterfield",
        "PersonB": "Hawkins",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Marjorie",
        "PersonB": "Edward",
        "RelationshipAtoB": "son",
        "RelationshipBtoA": "mother",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Marjorie",
        "PersonB": "Eliza",
        "RelationshipAtoB": "governess",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Marjorie",
        "PersonB": "Hawkins",
        "RelationshipAtoB": "butler",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
],
'scene_2_relationships' : 
[
    {
        "PersonA": "Mr. Fredrick Harrington",
        "PersonB": "Mrs. Harrington",
        "RelationshipAtoB": "spouse",
        "RelationshipBtoA": "husband",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mr. Fredrick Harrington",
        "PersonB": "Martha",
        "RelationshipAtoB": "housekeeper",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    },
    {
        "PersonA": "Mrs. Harrington",
        "PersonB": "Aiden Harrington",
        "RelationshipAtoB": "nephew",
        "RelationshipBtoA": "aunt",
        "CorrectChoice": "BtoA"
    },
    {
        "PersonA": "Mrs. Harrington",
        "PersonB": "Martha",
        "RelationshipAtoB": "housekeeper",
        "RelationshipBtoA": "employer",
        "CorrectChoice": "AtoB"
    }
]
}

def check_nested_dict(dictionary):
    for title, list in dictionary.items():
        for dict in list:
            if dict['RelationshipAtoB'] not in relationship_categories:
                print(f"Unknown relation: {dict['RelationshipAtoB']}")
            if dict['RelationshipBtoA'] not in relationship_categories:
                print(f"Unknown relation: {dict['RelationshipBtoA']}")