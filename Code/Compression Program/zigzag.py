import numpy as np


def zigzag(imageInput):

    height = 0
    Vertex = 0
    vertexMinimum = 0
    heightMinimum = 0
    vertexMaximum = imageInput.shape[0]
    HeightMaximum = imageInput.shape[1]
    i = 0

    output = np.zeros(vertexMaximum * HeightMaximum)

    # ----------------------------------

    while Vertex < vertexMaximum and height < HeightMaximum:

        if (height + Vertex) % 2 == 0:

            if Vertex == vertexMinimum:


                output[i] = imageInput[Vertex, height]

                if height == HeightMaximum:
                    Vertex = Vertex + 1
                else:
                    height = height + 1

                i = i + 1
            elif height == HeightMaximum - 1 and Vertex < vertexMaximum:



                output[i] = imageInput[Vertex, height]
                Vertex = Vertex + 1
                i = i + 1
            elif Vertex > vertexMinimum and height < HeightMaximum - 1:


                output[i] = imageInput[Vertex, height]
                Vertex = Vertex - 1
                height = height + 1
                i = i + 1
        else:


            if Vertex == vertexMaximum - 1 and height <= HeightMaximum - 1:

                output[i] = imageInput[Vertex, height]
                height = height + 1
                i = i + 1
            elif height == heightMinimum:



                output[i] = imageInput[Vertex, height]

                if Vertex == vertexMaximum - 1:
                    height = height + 1
                else:
                    Vertex = Vertex + 1

                i = i + 1
            elif Vertex < vertexMaximum - 1 and height > heightMinimum:


                output[i] = imageInput[Vertex, height]
                Vertex = Vertex + 1
                height = height - 1
                i = i + 1

        if Vertex == vertexMaximum - 1 and height == HeightMaximum - 1:



            output[i] = imageInput[Vertex, height]
            break


    return output


def inZ(input, vmax, hmax):


    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))

    i = 0

    # ----------------------------------

    while v < vmax and h < hmax:



        if (h + v) % 2 == 0:  # going up

            if v == vmin:

                # print(1)

                output[v, h] = input[i]

                if h == hmax:
                    v = v + 1
                else:
                    h = h + 1

                i = i + 1
            elif h == hmax - 1 and v < vmax:


                output[v, h] = input[i]
                v = v + 1
                i = i + 1
            elif v > vmin and h < hmax - 1:



                output[v, h] = input[i]
                v = v - 1
                h = h + 1
                i = i + 1
        else:



            if v == vmax - 1 and h <= hmax - 1:


                output[v, h] = input[i]
                h = h + 1
                i = i + 1
            elif h == hmin:



                output[v, h] = input[i]
                if v == vmax - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif v < vmax - 1 and h > hmin:



                output[v, h] = input[i]
                v = v + 1
                h = h - 1
                i = i + 1

        if v == vmax - 1 and h == hmax - 1:


            output[v, h] = input[i]
            break

    return output
