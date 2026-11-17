from transformers import pipeline

# Point this to whatever your new folder is named
classifier = pipeline("text-classification", model="./nlp_model") 

print(classifier("Flood in Pittsburgh, Pennsylvania"))
print(classifier("Massive earthquake just shook the city"))