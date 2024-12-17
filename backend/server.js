const express = require("express");
const cors = require("cors");
const comparisonRouter = require("./routes/comparison");

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// 路由
app.use("/api", comparisonRouter);

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
