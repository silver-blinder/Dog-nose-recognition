import { NextResponse } from "next/server";
import { writeFile } from "fs/promises";
import { spawn } from "child_process";
import path from "path";
import { mkdir } from "fs/promises";

async function ensureUploadDir() {
  try {
    await mkdir("./public/uploads", { recursive: true });
  } catch (error) {
    console.error("Failed to create uploads directory:", error);
  }
}

async function saveFile(file: File, filename: string) {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);
  const filepath = path.join("./public/uploads", filename);
  await writeFile(filepath, buffer);
  return filepath;
}

export async function POST(request: Request) {
  try {
    await ensureUploadDir();

    const formData = await request.formData();
    const images = formData.getAll("images") as File[];

    if (!images || images.length !== 2) {
      return NextResponse.json({ error: "需要上传两张图片" }, { status: 400 });
    }

    // 保存文件
    const image1Path = await saveFile(images[0], `${Date.now()}-1${path.extname(images[0].name)}`);
    const image2Path = await saveFile(images[1], `${Date.now()}-2${path.extname(images[1].name)}`);

    // 调用Python脚本进行比对
    return new Promise((resolve) => {
      const pythonProcess = spawn("python", ["./model/compare.py", image1Path, image2Path]);
      let result = "";

      pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error(`Error: ${data}`);
      });

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          resolve(NextResponse.json({ error: "比对过程出错" }, { status: 500 }));
          return;
        }

        try {
          const comparison = JSON.parse(result);
          resolve(NextResponse.json(comparison));
        } catch (error) {
          resolve(NextResponse.json({ error: "解析结果失败" }, { status: 500 }));
        }
      });
    });
  } catch (error) {
    console.error("处理失败:", error);
    return NextResponse.json({ error: "处理请求失败" }, { status: 500 });
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
