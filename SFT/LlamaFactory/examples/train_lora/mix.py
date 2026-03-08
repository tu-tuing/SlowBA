from safetensors.torch import load_file, save_file
import os
import shutil

def copy_visual_weights(source_dir, target_dir):
    """
    将source_dir中的lm_head.weight复制到target_dir
    """
    
    print(f"🔍 源目录: {source_dir}")
    print(f"🔍 目标目录: {target_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return False
        
    if not os.path.exists(target_dir):
        print(f"❌ 目标目录不存在: {target_dir}")
        return False
    
    # 找到源文件中的lm_head.weight
    source_files = [f for f in os.listdir(source_dir) if f.endswith(".safetensors")]
    target_files = [f for f in os.listdir(target_dir) if f.endswith(".safetensors")]
    
    print(f"源模型文件: {source_files}")
    print(f"目标模型文件: {target_files}")
    
    source_weight_file = None
    target_weight_file = None
    
    # 找到包含lm_head.weight的源文件
    for file in source_files:
        file_path = os.path.join(source_dir, file)
        try:
            weights = load_file(file_path)
            if "lm_head.weight" in weights:
                source_weight_file = file_path
                print(f"✅ 在 {file} 中找到 lm_head.weight")
                print(f"   形状: {weights['lm_head.weight'].shape}")
                break
        except Exception as e:
            print(f"❌ 读取 {file} 失败: {e}")
            continue
    
    # 找到目标文件（假设是第一个文件）
    if target_files:
        target_weight_file = os.path.join(target_dir, target_files[0])
        print(f"🎯 目标文件: {target_files[0]}")
    else:
        print("❌ 目标目录中没有 .safetensors 文件")
        return False
    
    if source_weight_file and target_weight_file:
        try:
            # 加载源权重
            print("🔄 加载源权重...")
            source_weights = load_file(source_weight_file)
            if "lm_head.weight" not in source_weights:
                print("❌ 源文件中没有找到 lm_head.weight")
                return False
                
            visual_weight = source_weights["lm_head.weight"]
            print(f"✅ 成功加载源权重，形状: {visual_weight.shape}")
            
            # 备份目标文件
            backup_path = target_weight_file + ".bak"
            shutil.copy2(target_weight_file, backup_path)
            print(f"📁 已备份目标文件到: {os.path.basename(backup_path)}")
            
            # 加载目标文件
            print("🔄 加载目标权重...")
            target_weights = load_file(target_weight_file)
            
            # 复制权重
            print("🔄 复制权重...")
            target_weights["lm_head.weight"] = visual_weight
            
            # 保存到目标文件
            print("💾 保存修改后的权重...")
            save_file(target_weights, target_weight_file)
            print(f"✅ 成功将 lm_head.weight 从 {os.path.basename(source_weight_file)} 复制到 {os.path.basename(target_weight_file)}")
            
            # 验证
            print("🔍 验证复制结果...")
            final_weights = load_file(target_weight_file)
            if "lm_head.weight" in final_weights:
                print(f"✅ 验证通过，权重形状: {final_weights['lm_head.weight'].shape}")
                return True
            else:
                print("❌ 验证失败：目标文件中没有找到 lm_head.weight")
                return False
                
        except Exception as e:
            print(f"❌ 复制过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("❌ 找不到源文件或目标文件")
        return False

# 使用示例
if __name__ == "__main__":
    success = copy_visual_weights(
        source_dir="/data2/thz/models/guir1/guir1/GUI-R1-3B",
        target_dir="/data2/thz/Qwen-VL-Series-Finetune/output/qwen2.5_lora_merge_new3"
    )
    
    if success:
        print("\n🎉 复制完成!")
    else:
        print("\n❌ 复制失败!")