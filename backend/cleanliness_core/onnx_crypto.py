"""
ONNX模型文件加密解密工具模块

使用 AES-256-GCM 算法进行高性能流式加密解密
支持大文件（>200MB）的流式处理
"""

import os
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from loguru import logger


# 加密文件魔数标识
MAGIC_BYTES = b"ZEM_ENCRYPTED_ONNX"
MAGIC_LEN = len(MAGIC_BYTES)

# AES-GCM参数
KEY_SIZE = 32  # AES-256
IV_SIZE = 12  # GCM推荐IV长度
TAG_SIZE = 16  # GCM认证标签长度

# 默认密钥（硬编码，可后续扩展为从配置文件读取）
# 注意：实际使用时应该使用更安全的密钥管理方式
# Keep the legacy key value so existing encrypted ONNX assets remain readable.
_DEFAULT_KEY = b"zem_particle_onnx_encryption_key_2025!"[:KEY_SIZE]
if len(_DEFAULT_KEY) < KEY_SIZE:
    _DEFAULT_KEY = _DEFAULT_KEY.ljust(KEY_SIZE, b"\x00")


def get_encryption_key() -> bytes:
    """
    获取加密密钥

    Returns:
        bytes: 32字节的AES-256密钥
    """
    # 可以后续扩展为从配置文件、环境变量等读取
    return _DEFAULT_KEY


def is_encrypted_file(file_path: str) -> bool:
    """
    检测文件是否为加密格式

    Args:
        file_path: 文件路径

    Returns:
        bool: True表示是加密文件，False表示未加密
    """
    if not os.path.exists(file_path):
        return False

    # 方法1: 检查文件扩展名
    if file_path.endswith(".enc.onnx"):
        return True

    # 方法2: 检查文件头魔数
    try:
        with open(file_path, "rb") as f:
            magic = f.read(MAGIC_LEN)
            return magic == MAGIC_BYTES
    except Exception as e:
        logger.warning(f"检查文件加密状态失败: {e}")
        return False


def encrypt_onnx_file(input_path: str, output_path: Optional[str] = None, chunk_size: int = 16 * 1024 * 1024) -> str:
    """
    加密ONNX文件

    Args:
        input_path: 输入ONNX文件路径
        output_path: 输出加密文件路径，如果为None则自动生成（添加.enc.onnx后缀）
        chunk_size: 流式处理的chunk大小（字节），默认16MB

    Returns:
        str: 输出文件路径

    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 加密失败
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if output_path is None:
        output_path = input_path + ".enc.onnx"

    key = get_encryption_key()
    iv = get_random_bytes(IV_SIZE)

    try:
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # 写入魔数和IV
            fout.write(MAGIC_BYTES)
            fout.write(iv)

            # 流式加密
            while True:
                chunk = fin.read(chunk_size)
                if not chunk:
                    break
                encrypted_chunk = cipher.encrypt(chunk)
                fout.write(encrypted_chunk)

            # 获取认证标签并写入
            tag = cipher.digest()
            fout.write(tag)

        logger.info(f"文件加密成功: {input_path} -> {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"加密文件失败: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError(f"加密失败: {e}")


def decrypt_onnx_to_bytes(file_path: str, chunk_size: int = 16 * 1024 * 1024) -> bytes:
    """
    解密ONNX文件到内存bytes（流式处理）

    Args:
        file_path: 加密文件路径
        chunk_size: 流式处理的chunk大小（字节），默认16MB

    Returns:
        bytes: 解密后的ONNX文件内容

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 解密失败或文件格式错误
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    key = get_encryption_key()

    try:
        with open(file_path, "rb") as f:
            # 读取并验证魔数
            magic = f.read(MAGIC_LEN)
            if magic != MAGIC_BYTES:
                raise ValueError(f"无效的加密文件格式: {file_path}")

            # 读取IV
            iv = f.read(IV_SIZE)
            if len(iv) != IV_SIZE:
                raise ValueError("文件格式错误: IV长度不正确")

            # 获取文件大小，计算加密数据长度（排除魔数、IV和tag）
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            encrypted_data_size = file_size - MAGIC_LEN - IV_SIZE - TAG_SIZE

            if encrypted_data_size <= 0:
                raise ValueError("文件格式错误: 加密数据长度为0")

            # 创建解密器
            cipher = AES.new(key, AES.MODE_GCM, nonce=iv)

            # 流式解密
            f.seek(MAGIC_LEN + IV_SIZE)
            decrypted_chunks = []

            remaining = encrypted_data_size
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                encrypted_chunk = f.read(read_size)
                if not encrypted_chunk:
                    break

                decrypted_chunk = cipher.decrypt(encrypted_chunk)
                decrypted_chunks.append(decrypted_chunk)
                remaining -= len(encrypted_chunk)

            # 读取并验证认证标签
            tag = f.read(TAG_SIZE)
            if len(tag) != TAG_SIZE:
                raise ValueError("文件格式错误: 认证标签长度不正确")

            # 验证认证标签（GCM会自动验证）
            try:
                cipher.verify(tag)
            except ValueError as e:
                raise ValueError(f"文件认证失败，可能被篡改: {e}")

            # 合并所有解密后的chunks
            result = b"".join(decrypted_chunks)

            logger.debug(f"文件成功解密到内存: {file_path}, 大小: {len(result)} 字节")
            return result

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"解密文件失败: {e}")
        raise ValueError(f"解密失败: {e}")


def load_onnx_model(file_path: str):
    """
    智能加载ONNX模型（自动检测加密/未加密）

    这是一个辅助函数，用于统一处理加密和未加密文件的加载

    Args:
        file_path: ONNX文件路径（加密或未加密）

    Returns:
        bytes或str: 如果是加密文件返回bytes，如果是未加密文件返回文件路径
    """
    if is_encrypted_file(file_path):
        return decrypt_onnx_to_bytes(file_path)
    else:
        return file_path
