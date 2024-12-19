import { NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import { createClient } from "@supabase/supabase-js";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

async function getNextFolderNumber(): Promise<string> {
  const { data: folders } = await supabase.storage.from("Dog noses").list();

  // 过滤出数字文件夹并获取最大编号
  const numbers = folders
    ?.map((folder) => {
      const match = folder.name.match(/^(\d{3})$/);
      return match ? parseInt(match[1]) : 0;
    })
    .filter((num) => !isNaN(num));

  const maxNumber = Math.max(0, ...(numbers || []));
  // 格式化为 3 位数字（001, 002 等）
  return String(maxNumber + 1).padStart(3, "0");
}

async function uploadImgToSupabase(file: File, isFirst: boolean, folderNumber: string) {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);

  const filename = `${folderNumber}/dog-nose-${isFirst ? "1" : "2"}${path.extname(file.name)}`;

  const { data, error } = await supabase.storage.from("Dog noses").upload(filename, buffer, {
    contentType: file.type,
    upsert: false,
  });
  if (error) {
    console.log(error);
    throw new Error("Failed to upload image to Supabase");
  }
  const {
    data: { publicUrl },
  } = supabase.storage.from("Dog noses").getPublicUrl(data.path);
  return publicUrl;
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const images = formData.getAll("images") as File[];

    if (!images || images.length !== 2) {
      return NextResponse.json({ error: "需要上传两张图片" }, { status: 400 });
    }

    const folderNumber = await getNextFolderNumber();

    // 保存文件
    const [image1Url, image2Url] = await Promise.all([
      uploadImgToSupabase(images[0], true, folderNumber),
      uploadImgToSupabase(images[1], false, folderNumber),
    ]);

    // 调用Python脚本进行比对
    return new Promise((resolve) => {
      const pythonProcess = spawn("python", ["./model/compare.py", image1Url, image2Url]);
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
