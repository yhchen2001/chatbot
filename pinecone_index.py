import os
from uuid import uuid4
import random
import time
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
from pdf_processing import pdf_to_df

class PineconeIndex():
	def __init__(self, pc, index_name):
		if index_name not in pc.list_indexes().names():
			pc.create_index(
				name=index_name,
				dimension=1536,
				metric='cosine',
				spec=PodSpec(
					environment='gcp-starter',
				)
			)
		while not pc.describe_index(index_name).status['ready']:
			continue
		self.index = pc.Index(index_name)

	def upsert_pdf(self, df, filename='', department=''):
		count, vectors = self.index.describe_index_stats()['total_vector_count'], []

		prev = ''
		for embedding, text in zip(df['embeddings'], df['text']):
			if len(vectors):
				vectors[-1]['metadata']['text'] += text

			vectors.append({
				'id': str(uuid4()),
				'values': embedding,
				'metadata': {
					'department': department,
					'filename': filename,
					'text': prev + text,
				},
			})

			prev = text

		self.index.upsert(vectors=vectors)
		count += len(vectors)
		while count != self.index.describe_index_stats()['total_vector_count']:
			continue
	
	def add_pdf(self, file_name):
		df = pdf_to_df(file_name)
		#print(df)
		self.upsert_pdf(df)

	def query(self, embedding, top_k=5, department=''):
		metadatas = [r['metadata'] for r in self.index.query(vector=embedding, top_k=top_k, include_metadata=True)['matches']]
		print(metadatas)

if __name__ == '__main__':
	load_dotenv()
	index_name = 'tsidgrp2'
	pci = PineconeIndex(Pinecone(os.getenv('PINECONE_API_KEY')), index_name)
	pci.upsert_pdf(
		{
			'text': ['a', 'b', 'c'],
			'embeddings': [[random.uniform(-1, 1) for _ in range(1536)] for __ in range(10)]
		}
	)
	pci.query([random.uniform(-1, 1) for _ in range(1536)])