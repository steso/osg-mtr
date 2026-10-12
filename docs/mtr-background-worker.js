self.onmessage = event => {
  const { data, dimensions, floor } = event.data;
  try {
    const [width, height, depth] = dimensions;
    const planeSize = width * height;
    const volumeSize = planeSize * depth;
    if (!width || !height || !depth || volumeSize > data.length) {
      throw new Error("The MTR dimensions do not match its voxel data.");
    }

    const output = new data.constructor(data.length);
    const total = data.length;
    const reportEvery = Math.max(1, Math.floor(total / 100));
    for (let offset = 0; offset < total; offset += volumeSize) {
      for (let z = 0; z < depth; z += 1) {
        for (let y = 0; y < height; y += 1) {
          for (let x = 0; x < width; x += 1) {
            const index = offset + z * planeSize + y * width + x;
            const center = data[index];
            if (!Number.isFinite(center) || center <= floor) {
              output[index] = 0;
              continue;
            }

            let supportedNeighbors = 0;
            for (let zSide = 0; zSide < 2; zSide += 1) {
              const neighborZ = z + (zSide === 0 ? -1 : 1);
              if (neighborZ < 0 || neighborZ >= depth) continue;
              for (let ySide = 0; ySide < 2; ySide += 1) {
                const neighborY = y + (ySide === 0 ? -1 : 1);
                if (neighborY < 0 || neighborY >= height) continue;
                for (let xSide = 0; xSide < 2; xSide += 1) {
                  const neighborX = x + (xSide === 0 ? -1 : 1);
                  if (neighborX < 0 || neighborX >= width) continue;
                  const value = data[offset + neighborZ * planeSize + neighborY * width + neighborX];
                  if (Number.isFinite(value) && value > floor) {
                    supportedNeighbors += 1;
                  }
                }
              }
            }
            output[index] = supportedNeighbors >= 8 ? center : 0;
          }
          const completed = offset + z * planeSize + (y + 1) * width;
          if (completed % reportEvery < width) {
            self.postMessage({ type: "progress", percent: Math.min(99, Math.floor(completed / total * 100)) });
          }
        }
      }
    }
    self.postMessage({ type: "complete", buffer: output.buffer }, [output.buffer]);
  } catch (error) {
    self.postMessage({ type: "error", message: error.message || String(error) });
  }
};
