import { spawn } from "node:child_process";
import path from "node:path";
import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

async function getAllDogNoseImages() {
  const { data: dogsData, error } = await supabase.from("dogs").select("id, image_url");

  if (error) {
    throw new Error("获取狗鼻子图片失败");
  }

  return dogsData || [];
}

async function uploadTempImage(file: File) {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);

  const filename = `temp-search-${Date.now()}.jpg`;

  const { data, error } = await supabase.storage.from("dog-noses").upload(filename, buffer, {
    contentType: file.type,
    upsert: false,
  });

  if (error) {
    console.log(error);
    throw new Error("上传临时图片到 Supabase 失败");
  }

  const {
    data: { publicUrl },
  } = supabase.storage.from("dog-noses").getPublicUrl(data.path);

  return { publicUrl, path: filename };
}

async function compareWithExistingImages(newImageUrl: string) {
  // 获取所有已存储的狗鼻子图片
  const existingDogs = await getAllDogNoseImages();

  // 如果数据库为空，直接返回没有匹配
  if (existingDogs.length === 0) {
    return { matches: [], bestMatch: null };
  }

  // 对每个现有图片进行比较
  const comparisonPromises = existingDogs.map(async (dog) => {
    return new Promise((resolve) => {
      const pythonProcess = spawn("python", ["./model/compare.py", newImageUrl, dog.image_url]);
      let result = "";

      pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error(`Error: ${data}`);
      });

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          resolve({ dogId: dog.id, is_same_dog: false, confidence: 0 });
          return;
        }

        try {
          const comparison = JSON.parse(result);
          resolve({
            dogId: dog.id,
            is_same_dog: comparison.is_same_dog,
            confidence: comparison.confidence,
          });
        } catch {
          resolve({ dogId: dog.id, is_same_dog: false, confidence: 0 });
        }
      });
    });
  });

  const comparisons = await Promise.all(comparisonPromises);

  // 找出匹配的狗
  const matches = comparisons.filter((comp: any) => comp.is_same_dog);

  // 找出最佳匹配（置信度最高的）
  let bestMatch: any = null;
  if (matches.length > 0) {
    bestMatch = matches.reduce((prev: any, current: any) =>
      prev.confidence > current.confidence ? prev : current
    );

    // 获取最佳匹配的完整信息
    if (bestMatch) {
      const { data, error } = await supabase
        .from("dogs")
        .select("*")
        .eq("id", bestMatch.dogId)
        .single();

      if (!error && data) {
        bestMatch.dogInfo = data;
      }
    }
  }

  return { matches, bestMatch };
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const noseImage = formData.get("noseImage") as File;

    if (!noseImage) {
      return NextResponse.json({ error: "需要上传狗鼻子图片" }, { status: 400 });
    }

    // 上传临时图片
    const { publicUrl, path: tempPath } = await uploadTempImage(noseImage);

    // 与已有图片比较
    const comparisonResult = await compareWithExistingImages(publicUrl);

    // 删除临时图片
    await supabase.storage.from("dog-noses").remove([tempPath]);

    // 如果有匹配，返回匹配信息
    if (comparisonResult.bestMatch && comparisonResult.bestMatch.is_same_dog) {
      return NextResponse.json({
        found: true,
        dogInfo: comparisonResult.bestMatch.dogInfo,
        confidence: comparisonResult.bestMatch.confidence,
      });
    }

    // 如果没有匹配，返回未找到
    return NextResponse.json({
      found: false,
      message: "未找到匹配的狗狗",
    });
  } catch (error) {
    console.error("处理失败:", error);
    return NextResponse.json({ error: "处理请求失败" }, { status: 500 });
  }
}
