import argparse
import zlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import sympy
from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class Chunk:
    name: bytes
    data: bytes

    @property
    def length(self) -> int:
        return len(self.data)

    def to_bytes(self) -> bytes:
        """Serializuje chunk: [długość (4B)] + [typ (4B)] + [dane] + [CRC32 (4B)]."""
        length_bytes = self.length.to_bytes(4, "big")
        crc = zlib.crc32(self.name + self.data).to_bytes(4, "big")
        return length_bytes + self.name + self.data + crc

    def __repr__(self) -> str:
        return f"Chunk(name={self.name.decode('ascii', errors='replace')}, length={self.length})"


class PNGParser:
    @staticmethod
    def parse(file_bytes: bytes) -> list[Chunk]:
        if not file_bytes.startswith(PNG_SIGNATURE):
            raise ValueError("Plik nie ma poprawnej sygnatury formatu PNG.")

        chunks = []
        offset = len(PNG_SIGNATURE)
        total_len = len(file_bytes)

        while offset < total_len:
            if offset + 8 > total_len:
                break
            length = int.from_bytes(file_bytes[offset : offset + 4], "big")
            name = file_bytes[offset + 4 : offset + 8]
            data_start = offset + 8
            data_end = data_start + length
            data = file_bytes[data_start:data_end]

            chunks.append(Chunk(name=name, data=data))
            offset = data_end + 4

        return chunks

    @staticmethod
    def rebuild(chunks: list[Chunk]) -> bytes:
        out = bytearray(PNG_SIGNATURE)
        for chunk in chunks:
            out.extend(chunk.to_bytes())
        return bytes(out)


class RSAEngine:
    def __init__(self, bits: int = 1024):
        self.bits = bits
        self.modulus, self.public_key, self.private_key = self._generate_keys()

    def _generate_keys(self):
        half_bits = self.bits // 2
        p = sympy.randprime(pow(2, half_bits - 1), pow(2, half_bits))
        q = sympy.randprime(pow(2, half_bits - 1), pow(2, half_bits))
        while p == q:
            q = sympy.randprime(pow(2, half_bits - 1), pow(2, half_bits))

        n = p * q
        totient = (p - 1) * (q - 1)

        e = 65537
        if sympy.gcd(e, totient) != 1:
            while True:
                e = sympy.randprime(3, totient)
                if sympy.gcd(e, totient) == 1:
                    break

        d = pow(e, -1, totient)
        return n, e, d

    @property
    def key_size(self) -> int:
        return (self.modulus.bit_length() + 7) // 8

    def encrypt_data(self, data: bytes, block_size: int = 1) -> bytes:
        encrypted = bytearray()
        k_size = self.key_size
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            val = int.from_bytes(block, "big")
            enc_val = pow(val, self.public_key, self.modulus)
            encrypted.extend(enc_val.to_bytes(k_size, "big"))
        return bytes(encrypted)

    def decrypt_data(self, enc_data: bytes, block_size: int = 1) -> bytes:
        decrypted = bytearray()
        k_size = self.key_size
        for i in range(0, len(enc_data), k_size):
            chunk = enc_data[i : i + k_size]
            val = int.from_bytes(chunk, "big")
            dec_val = pow(val, self.private_key, self.modulus)
            decrypted.extend(dec_val.to_bytes(block_size, "big"))
        return bytes(decrypted)


def process_image(input_path: str, key_bits: int = 1024):
    print(f"Generowanie kluczy RSA ({key_bits} bitów)...")
    rsa = RSAEngine(bits=key_bits)

    with open(input_path, "rb") as f:
        file_bytes = f.read()

    chunks = PNGParser.parse(file_bytes)

    idat_encrypted_chunks = []
    idat_decrypted_chunks = []

    for chunk in chunks:
        if chunk.name == b"IDAT":
            enc_payload = rsa.encrypt_data(chunk.data, block_size=1)
            dec_payload = rsa.decrypt_data(enc_payload, block_size=1)
            idat_encrypted_chunks.append(Chunk(name=chunk.name, data=enc_payload))
            idat_decrypted_chunks.append(Chunk(name=chunk.name, data=dec_payload))
        else:
            idat_encrypted_chunks.append(Chunk(name=chunk.name, data=chunk.data))
            idat_decrypted_chunks.append(Chunk(name=chunk.name, data=chunk.data))

    decomp_encrypted_chunks = []
    decomp_decrypted_chunks = []

    for chunk in chunks:
        if chunk.name == b"IDAT":
            raw_pixels = zlib.decompress(chunk.data)
            enc_pixels = rsa.encrypt_data(raw_pixels, block_size=4)
            dec_pixels = rsa.decrypt_data(enc_pixels, block_size=4)

            decomp_encrypted_chunks.append(
                Chunk(name=chunk.name, data=zlib.compress(enc_pixels))
            )
            decomp_decrypted_chunks.append(
                Chunk(name=chunk.name, data=zlib.compress(dec_pixels))
            )
        else:
            decomp_encrypted_chunks.append(Chunk(name=chunk.name, data=chunk.data))
            decomp_decrypted_chunks.append(Chunk(name=chunk.name, data=chunk.data))

    with open("decrypted.png", "wb") as f:
        f.write(PNGParser.rebuild(idat_decrypted_chunks))

    with open("decrypted_decompressed.png", "wb") as f:
        f.write(PNGParser.rebuild(decomp_decrypted_chunks))

    img_orig = Image.open(input_path)
    img_dec1 = Image.open("decrypted.png")
    img_dec2 = Image.open("decrypted_decompressed.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_orig)
    axes[0].set_title("Oryginał")
    axes[1].imshow(img_dec1)
    axes[1].set_title("Odszyfrowany (IDAT raw)")
    axes[2].imshow(img_dec2)
    axes[2].set_title("Odszyfrowany (zlib stream)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo szyfrowania struktury PNG chunków IDAT algorytmem RSA."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="pp0n6a08.png",
        help="Ścieżka do pliku PNG (domyślnie: pp0n6a08.png)",
    )
    parser.add_argument(
        "--bits",
        "-b",
        type=int,
        default=512,
        help="Długość klucza RSA w bitach (mniejsza wartość = szybsze demo, np. 512 lub 1024)",
    )
    args = parser.parse_args()

    process_image(args.file, key_bits=args.bits)
