import numpy as np
import cv2
import math
def get_bandwidth():
    bdwidth = [1,2,3,4,5,6]
    for i in range(100):
        yield bdwidth[i%6]

def decompressed_block(list_block, bs = 4):
    block = np.zeros((bs,bs), dtype = np.uint8)

    H, L = list_block[0], list_block[1]
    ptr = 2

    for i in range(bs):
        for j in range(bs):
            if list_block[ptr] == 1:
                block[i][j] = H
            else:
                block[i][j] = L
            ptr += 1
    
    return block

def decompress_image(img_cmp, bs = 4):
    m, n = img_cmp[0], img_cmp[1]

    img_final = np.zeros((m,n), dtype = np.uint8)
    img_cmp = img_cmp[2:]

    b = bs*bs + 2

    ptr = 0
    row = m//bs
    col = n//bs

    for i in range(row):
        for j in range(col):
            block = decompressed_block(img_cmp[ptr: ptr + (2 + bs*bs)], bs)
            img_final[bs*i:bs*i+bs, bs*j:bs*j+bs] = block
            ptr += (2 + bs*bs)
    return img_final

class Block_Truncation:
    def __init__(self, img, block_size = 4):
        bs = block_size
        m, n = img.shape
        rp = 0
        cp = 0
        if m%bs != 0:
            rp = bs-(m%bs)
        if n%bs != 0:
            cp = bs-(n%bs)
        
        temp = np.zeros((m+rp, n+cp), dtype = np.uint8)
        temp[0:m, 0:n] = img

        m, n = temp.shape
        row = m//bs
        col = n//bs

        compressed = [m, n]
        for i in range(row):
            for j in range(col):
                block = img[bs*i:bs*i+bs, bs*j:bs*j+bs]
                list_block = self.get_compressed_block(block)
                compressed.extend(list_block)
        
        self.compressed = compressed
    
    def calc_HL(self, block):
        m, n = block.shape
        sigma = np.std(block)
        mu = np.mean(block)
        epsilon = 0.0001
        temp = block.copy()
        temp[block >= mu] = 1
        temp[block < mu] = 0

        q = np.sum(temp)

        H = int(round(mu + sigma * math.sqrt((m*n - q)/(q+epsilon))))
        L = int(round(mu - sigma * math.sqrt(q/(m*n - q + epsilon))))

        if H > 255:
            H =255
        if L < 0:
            L = 0

        return H, L, mu

    def get_compressed_block(self, block):
        H, L, mu = self.calc_HL(block)
        out = block.copy()
        out[block >= mu] = 1 
        out[block < mu] = 0

        output = [H, L]
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                output.append(out[i][j])

        return output

if __name__ == '__main__':
    itr = get_bandwidth()
    cam = cv2.VideoCapture(0)

    while True:
        flag = None
        img_cmp = None
        # bw = next(itr)
        bw = 4
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if bw < 5:
            img_cmp = Block_Truncation(frame, 16).compressed 
        
        out_frame = decompress_image(img_cmp, 16)
        out_frame = cv2.blur(out_frame, (7, 7))    #Tweak in this parameter to adjust blur (is always odd - (o,o))
        cv2.imshow('Video Feed', frame)
        cv2.imshow('Output', out_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()