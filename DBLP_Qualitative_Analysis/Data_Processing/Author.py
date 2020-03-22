import numpy as np
from collections import defaultdict


class Author:
    
    def __init__(self, data):
        
        self.id = data['id']
        self.name = data['name']
        self.org = data['org']
        
        self.fields = {}
        self.contents = defaultdict(list)
        self.venues = defaultdict(list)
        self.outcitations = defaultdict(list)
        self.incitations = defaultdict(list)        
        
        self.embeddings = {}
        self.cite_edgellhs = {}
        self.cite_straranks = {}
        self.cite_atts = defaultdict(dict)
        self.cite_util_histories = {}       
        self.pub_edgellhs = {}
        self.pub_straranks = {}
        self.pub_atts = defaultdict(dict)
        self.pub_util_histories = {}
                
        
    def update_contents(self, year, content, field, venue):
        
        self.contents[year].append(content)
        self.venues[year].append(venue)
        
        if year in self.fields: self.fields[year] += field.astype(np.float64)
        else: self.fields[year] = field.astype(np.float64)
            
            
    def update_fields(self):
        
        years = sorted(self.fields)
        fields = [self.fields[year] for year in years]
        counts = [len(self.contents[year]) for year in years]
        
        for i in range(1,len(years)):
            fields[i] += fields[i-1]
            counts[i] += counts[i-1]
        
        for i, year in enumerate(years):
            self.fields[year] = (fields[i]/counts[i]).astype(np.float32)        
            
            
    # I cite inauthor's inyear's content in outyear
    def update_outcitations(self, outcitations):
        
        for outyear, inyear, inauthor in outcitations:
            self.outcitations[outyear].append((inyear, inauthor))
            
    
    # Outauthor cites my inyear's content in outyear
    def update_incitations(self, incitations):
        
        for outyear, inyear, outauthor in incitations:
            self.incitations[outyear].append((inyear, outauthor))
    
    
    def update_embeddings(self, year, emb):
        
        self.embeddings[year] = emb
        
        
    def update_cite_edgellhs(self, year, cite_edgellhs):
        
        self.cite_edgellhs[year] = cite_edgellhs
        
        
    def update_cite_stradiss(self, cite_stradiss):
        
        self.cite_stradiss = cite_stradiss
        
        
    def update_cite_straranks(self):
        
        for year, cite_stradis in self.cite_stradiss.items():
            self.cite_straranks[year] = np.argsort(-cite_stradis)
        
        
    def update_cite_alphas(self, cite_alphas):
        
        self.cite_alphas = cite_alphas
        
        
    def update_cite_atts(self, cite_atts):
        
        for year, content, cite_att in cite_atts:
            self.cite_atts[year][content] = cite_att
            
            
    def update_cite_utilities(self, cite_utilities):
        
        self.cite_utilities = cite_utilities
    
        
    def update_cite_util_histories(self):
        
        start, end = 2000, 2019
        cite_util_histories = np.zeros((end-start, end-start))
        
        for year, contents in self.contents.items():
            for content in contents:
                if content in self.cite_utilities:
                    for end_year, util in self.cite_utilities[content]:
                        cite_util_histories[year-start][end_year-start] += util
                        
        for year in range(1, end-start):
            cite_util_histories[:,year] += cite_util_histories[:,year-1]
            
        for year in sorted(self.cite_straranks.keys()):
            self.cite_util_histories[year] = (self.cite_straranks[year][0], len(self.contents[year]), cite_util_histories[year-start], self.cite_alphas[year])        
            
            
    def update_pub_edgellhs(self, year, pub_edgellhs):
        
        self.pub_edgellhs[year] = pub_edgellhs
        
        
    def update_pub_stradiss(self, pub_stradiss):
        
        self.pub_stradiss = pub_stradiss
        
        
    def update_pub_straranks(self):
        
        for year, pub_stradis in self.pub_stradiss.items():
            self.pub_straranks[year] = np.argsort(-pub_stradis)
        
        
    def update_pub_alphas(self, pub_alphas):
        
        self.pub_alphas = pub_alphas
        
        
    def update_pub_atts(self, pub_atts):
        
        for year, content, pub_att in pub_atts:
            self.pub_atts[year][content] = pub_att
            
            
    def update_pub_utilities(self, pub_utilities):
        
        self.pub_utilities = pub_utilities
        
        
    def update_pub_util_histories(self):
        
        start, end = 2000, 2019
        pub_util_histories = np.zeros((end-start, end-start))
        
        for year, contents in self.contents.items():
            for content in contents:
                if content in self.pub_utilities:
                    for end_year, util in self.pub_utilities[content]:
                        pub_util_histories[year-start][end_year-start] += util
                        
        for year in range(1, end-start):
            pub_util_histories[:,year] += pub_util_histories[:,year-1]
            
        for year in sorted(self.pub_straranks.keys()):
            self.pub_util_histories[year] = (self.pub_straranks[year][0], len(self.contents[year]), pub_util_histories[year-start], self.pub_alphas[year])