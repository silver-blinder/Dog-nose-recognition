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

async function getNextFolderNumber(): Promise<string> {
  const { data: folders } = await supabase.storage.from("dog-noses").list();

  // 过滤出数字文件夹并获取最大编号
  const numbers = folders
    ?.map((folder) => {
      const match = folder.name.match(/^(\d{3})$/);
      return match ? Number.parseInt(match[1]) : 0;
    })
    .filter((num) => !Number.isNaN(num));

  const maxNumber = Math.max(0, ...(numbers || []));
  // 格式化为 3 位数字（001, 002 等）
  return String(maxNumber + 1).padStart(3, "0");
}

async function uploadImgToSupabase(file: File, folderNumber: string) {
  const bytes = await file.arrayBuffer();
  const buffer = Buffer.from(bytes);

  const filename = `${folderNumber}/dog-nose${path.extname(file.name)}`;

  const { data, error } = await supabase.storage.from("dog-noses").upload(filename, buffer, {
    contentType: file.type,
    upsert: false,
  });
  if (error) {
    console.log(error);
    throw new Error("上传图片到 Supabase 失败");
  }
  const {
    data: { publicUrl },
  } = supabase.storage.from("dog-noses").getPublicUrl(data.path);
  return publicUrl;
}

async function saveDogInfoToDatabase(dogInfo: any, imageUrl: string) {
  const { data, error } = await supabase
    .from("dogs")
    .insert({
      name: dogInfo.name,
      gender: dogInfo.gender,
      age: dogInfo.age,
      breed: dogInfo.breed,
      image_url: imageUrl,
      created_at: new Date().toISOString(),
    })
    .select();

  if (error) {
    throw new Error("保存狗的信息到数据库失败");
  }

  return data?.[0] || null;
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

    // 获取狗的信息
    const name = formData.get("name") as string;
    const gender = formData.get("gender") as string;
    const age = formData.get("age") as string;
    const breed = formData.get("breed") as string;

    // 先上传图片
    const folderNumber = await getNextFolderNumber();
    const imageUrl = await uploadImgToSupabase(noseImage, folderNumber);

    // 与已有图片比较
    const comparisonResult = await compareWithExistingImages(imageUrl);

    // 如果有匹配，返回匹配信息
    if (comparisonResult.bestMatch && comparisonResult.bestMatch.is_same_dog) {
      return NextResponse.json({
        is_duplicate: true,
        message: "该狗已存在于数据库中",
        matches: comparisonResult.matches,
        bestMatch: comparisonResult.bestMatch,
      });
    }

    // 如果没有匹配，保存狗的信息到数据库
    const dogInfo = await saveDogInfoToDatabase({ name, gender, age, breed }, imageUrl);

    return NextResponse.json({
      is_duplicate: false,
      message: "新狗已成功添加到数据库",
      dogInfo,
    });
  } catch (error) {
    console.error("处理失败:", error);
    return NextResponse.json({ error: "处理请求失败" }, { status: 500 });
  }
}
