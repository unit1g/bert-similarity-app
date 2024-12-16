const express = require("express");
const cors = require("cors");
const { pipeline } = require("@xenova/transformers");

const app = express();
const port = process.env.PORT || 3004;

// Middleware
app.use(cors());
app.use(express.json());

// Load the BERT model (using @xenova/transformers for Node.js)
let model = null;
(async () => {
  console.log("Loading model...");
  model = await pipeline("feature-extraction", "Xenova/bert-base-uncased");
  console.log("Model loaded.");
})();

// Helper function to calculate cosine similarity
function cosineSimilarity(vec1, vec2) {
  const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const magnitude1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const magnitude2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitude1 * magnitude2);
}

// Endpoint to calculate similarity
app.post("/api/similarity", async (req, res) => {
  const { text1, text2 } = req.body;

  if (!text1 || !text2) {
    return res.status(400).json({ error: "Both text1 and text2 are required" });
  }

  if (!model) {
    return res.status(503).json({ error: "Model is not loaded yet. Please try again later." });
  }

  try {
    const emb1 = await model(text1);
    const emb2 = await model(text2);

    // Use the first vector from the output for similarity calculation
    const similarity = cosineSimilarity(emb1[0], emb2[0]);
    res.json({ similarity: similarity * 100 }); // Return similarity as a percentage
  } catch (error) {
    console.error("Error calculating similarity:", error);
    res.status(500).json({ error: "Failed to calculate similarity" });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
