import numpy as np
from collections import defaultdict


class Venue:
    
    def __init__(self, data):
        
        self.id = data['id']
        self.name = data['name']
        
        self.fields = {}
        self.contents = defaultdict(list)
        self.authors = defaultdict(list)
        self.embeddings = {}
        
        
    def update_contents(self, year, content, field, authors):
        
        self.contents[year].append(content)
        self.authors[year].append(authors)
        
        if year in self.fields: self.fields[year] += field.astype(np.float64)
        else: self.fields[year] = field.astype(np.float64)
            
            
    def update_authors(self):
        
        updated_authors = defaultdict(dict)
        for year, authors in self.authors.items():
            
            authors = np.concatenate(authors)
            authors, counts = np.unique(authors, return_counts=True)
            
            for author, count in zip(authors, counts):
                updated_authors[year][author] = count
                
        self.authors = updated_authors
        
        
    def update_fields(self):
        
        years = sorted(self.fields)
        fields = [self.fields[year] for year in years]
        counts = [len(self.contents[year]) for year in years]
        
        for i in range(1,len(years)):
            fields[i] += fields[i-1]
            counts[i] += counts[i-1]
        
        for i, year in enumerate(years):
            self.fields[year] = (fields[i]/counts[i]).astype(np.float32)
            
    
    def update_embeddings(self, year, emb):
        
        self.embeddings[year] = emb