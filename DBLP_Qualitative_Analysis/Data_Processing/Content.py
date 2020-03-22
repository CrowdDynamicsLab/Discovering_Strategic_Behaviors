import numpy as np
from collections import defaultdict


class Content:
    
    def __init__(self, data):
        
        self.id = data['id']
        self.title = data['title']
        self.year = data['year']
        self.fos = data['fos']
        self.venue = data['venue']
        self.authors = data['authors']        
        self.outcitations = data['outcitations']
        self.incitations = defaultdict(list)
        
        
    def update_incitations(self, incitations):
        
        for incitation, year in incitations:
            self.incitations[year].append(incitation)
        
    
    def update_field(self, field):
        
        self.field = field
        
        
    def update_embedding(self, emb):
        
        self.embedding = emb
        
        
    def update_cite_edgellhs(self, cite_edgellhs):
        
        self.cite_edgellhs = cite_edgellhs
        
        
    def update_cite_atts(self, cite_atts):
        
        self.cite_atts = cite_atts
    
    
    def update_cite_stradis(self, cite_stradis):
        
        self.cite_stradis = cite_stradis
    
    
    def update_pub_edgellhs(self, pub_edgellhs):
        
        self.pub_edgellhs = pub_edgellhs
    
            
    def update_pub_atts(self, pub_atts):
        
        self.pub_atts = pub_atts           
            

    def update_pub_stradis(self, pub_stradis):
        
        self.pub_stradis = pub_stradis
        
        
    def split_utility(self, strategy):
        
        if strategy=='pub': atts=self.pub_atts
        elif strategy=='cite': atts=self.cite_atts
        
        author_year_utility = defaultdict(list)
        for author,_,_ in self.authors:
            for year in sorted(self.incitations):
                author_year_utility[author].append((year, atts[author]*len(self.incitations[year])/2))
        
        return author_year_utility