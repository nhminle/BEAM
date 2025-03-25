import evaluate

# Load BLEURT
bleurt = evaluate.load("bleurt")

# Test pair: identical strings should yield a high score.
prediction = "This is a test."
reference = "This is a test."

result = bleurt.compute(predictions=[prediction], references=[reference])
score = result["scores"][0]
print("BLEURT score for identical texts:", score)
