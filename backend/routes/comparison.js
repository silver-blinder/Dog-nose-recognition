const express = require("express");
const router = express.Router();
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");

// 配置文件上传
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  },
});

const upload = multer({ storage: storage });

// 处理两张图片上传和比对
router.post("/compare", upload.array("images", 2), async (req, res) => {
  try {
    if (!req.files || req.files.length !== 2) {
      return res.status(400).json({ error: "需要上传两张图片" });
    }

    const image1Path = req.files[0].path;
    const image2Path = req.files[1].path;

    // 调用 Python 脚本进行比对
    const pythonProcess = spawn("python", ["model/compare.py", image1Path, image2Path]);

    let result = "";

    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Error: ${data}`);
    });

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        return res.status(500).json({ error: "比对过程出错" });
      }

      const comparison = JSON.parse(result);
      res.json(comparison);
    });
  } catch (error) {
    res.status(500).json({ error: "服务器错误" });
  }
});

module.exports = router;
