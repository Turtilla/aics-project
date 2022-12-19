from abc import ABC, abstractmethod
from itertools import permutations, product
from pprint import pprint

import nltk


class RelationFilter(ABC):
    @abstractmethod
    def has_relation(self, tokenized_caption: list[str]) -> bool:
        pass

class POSRelationFilter(RelationFilter):
    def has_relation(self, tokenized_caption: list[str]) -> bool:
        # selecting only the captions that include verbs or prepositions (relation words) by Maria
        annotated_caption = nltk.pos_tag(tokenized_caption, tagset='universal')

        va_counter = 0  # for seeing if there is a verb or an adposition in the description
        for tagged_word in annotated_caption:
            if tagged_word[1] == 'VERB':
                va_counter += 1
            elif tagged_word[1] == 'ADP':
                va_counter += 1
        
        return va_counter > 0


# relations and rules from https://github.com/GU-CLASP/spatial_relations_vectors_sltc2018
class RuleBasedRelationFilter(RelationFilter):
    def __init__(self) -> None:
        super().__init__()

        # Landau English prepositions
        self.en_preps = [
            # simple spatial relations
            'at', 'on', 'in', 'on', 'off',
            'out', 'by', 'from', 'to',
            'up', 'down', 'over', 'under',
            'with', ('within', 'with in'), ('without', 'with out'), 'near',
            'neadby', 'into', ('onto', 'on to'), 'toward',
            'through', 'throughout', 'underneath', 'along',
            'across', ('among', 'amongst'), 'against', 'around',
            'about', 'above', ('amid', 'amidst'), 'before',
            'behind', 'below', 'beneath', 'between',
            'beside', 'outside', 'inside', ('alongside', 'along side'),
            'via', 'after', 'upon', 
            # compounds
            ('top', 'on top of'), ('between', 'in between'), ('right', 'to the right of'), ('parallel', 'parallel to'),
            ('back', 'in back of'), ('left', 'to the left of'), ('side', 'to the side'), ('perpendicular', 'perpendicular to'),
            ('front', 'in front of'),
            # temporal only
            'during', 'since', 'until', 'ago',
            # intransitivies (+ additional variations)
            'here', 'outward', ('backward', 'backwards'), ('south' , 'south of'),
            'there', ('afterward', 'afterwards'), 'away', ('east', 'east of'),
            'upward', 'upstairs', 'apart', ('west', 'west of'),
            'downward', 'downstairs', 'together', 'left',
            'inward', 'sideways', ('north', 'north of'), 'right',
        ]

        # Herskovits projective_terms
        self.en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'to', 'by'] for w2 in ['left', 'right'] ]
        self.en_preps += [(w2, w1+' the '+w2+' side of')      for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
        self.en_preps += [(w2, w1+' the '+w2+' hand side of') for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
        self.en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['front', 'back', 'side']]
        self.en_preps += [(w1, 'in '+w1+' of')                for w1 in ['front', 'back']]
        self.en_preps += [(w1,)                               for w1 in ['before', 'behind']]
        self.en_preps += [(w1, w1+' of')                      for w1 in ['left', 'right', 'back']]
        self.en_preps += [(w1,)                               for w1 in ['above', 'below']]
        self.en_preps += [(w1,)                               for w1 in ['over', 'under']]
        self.en_preps += [(w2, w1+' the '+w2+' of')           for w1 in ['at', 'on', 'in', 'by'] for w2 in ['top', 'bottom']]
        self.en_preps += [(w2, w1+' '+w2+' of')               for w1 in ['on'] for w2 in ['top']]

        # missing items?
        self.en_preps += [('next', 'next to')]

        # missing odd variations
        self.en_preps += [('front', 'on the front of', 'on front of')]
        self.en_preps += [('left', 'in the left of', 'in left of'),('right', 'in the right of', 'in right of'),]

        # missing 'the'
        self.en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'to', 'by'] for w2 in ['left', 'right'] ]
        self.en_preps += [(w2, w1+' '+w2+' side of')      for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
        self.en_preps += [(w2, w1+' '+w2+' hand side of') for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['left', 'right']]
        self.en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'in', 'to', 'by'] for w2 in ['front', 'back', 'side']]
        self.en_preps += [(w2, w1+' '+w2+' of')           for w1 in ['at', 'on', 'in', 'by'] for w2 in ['top', 'bottom']]

        # compositional variation
        self.en_preps += [
            (w2+'_'+w3, w1+_the_+w2+_and_+w3+' of')
            for w1 in ['at', 'on', 'in', 'to', 'by', 'to']
            for _the_ in [' ', ' the ']
            for _and_ in [' ', ' and ']
            for x, y in permutations([
            ['upper', 'lower'],
                ['left', 'right',],
                ['front', 'back',],
                ['top', 'bottom'],
                ['before', 'behind'],
                ['above', 'over', 'under', 'below', ],
                #['next', 'close', 'far']
            ], 2)
            for w2, w3 in product(x,y)
        ]

        # fix the tuple types
        self.en_preps = [(w,) if not isinstance(w, tuple) else w for w in self.en_preps]

        # This will create a ditionary of preposition variations to a simple tocken
        self.composit2simple = dict()
        self.composit2simple.update({w_alt: w[0] for w in self.en_preps for w_alt in w})
        self.composit2simple.update({w: w        for w in self.composit2simple.values()})

    def has_relation(self, tokenized_caption: list[str]) -> bool:
        for relation in self.composit2simple:
            tokenized_relation = nltk.word_tokenize(relation)
            if any(tokenized_caption[idx: idx + len(tokenized_relation)] == tokenized_relation for idx in range(len(tokenized_caption) - len(tokenized_relation) + 1)):
                return True
        return False
