import numpy as np

class FaceSet:
    faces = []
    ref_images = []
    embedding_average = 'None'
    embeddings_backup = None

    def __init__(self):
        self.faces = []
        self.ref_images = []
        self.embeddings_backup = None

    def AverageEmbeddings(self):
        if len(self.faces) > 1 and self.embeddings_backup is None:
            self.embeddings_backup = self.faces[0]['embedding']
            embeddings = [face.embedding for face in self.faces]

            self.faces[0]['embedding'] = np.mean(embeddings, axis=0)
            # try median too?
