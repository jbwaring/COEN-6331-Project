#include <stdio.h>
#include <bitset>
#include <iostream>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <cstring> // for memcpy

#define DEBUG 1
extern "C"
{

    void print_bits(unsigned int num)
    {
        // You can use the bitset class to directly convert the unsigned int to its binary representation
        std::bitset<32> binary_representation(num);

        // Print the binary representation
        std::cout << binary_representation << "("
                  << "0x" << std::hex << num << ") ";
    }

    int sumWord(unsigned int num)
    {
        std::bitset<32> binary_representation(num);
        return binary_representation.count();
    }

    void printWord(vBool32 a)
    {
        for (int i = 0; i < 4; i++)
        {
            print_bits(a[i]);
        }
    }

    void printHV(vBool32 *a, int size)
    {
        for (int i = 0; i < size; i++)
        {
            printWord(a[i]);
        }
        std::cout << std::endl;
    }
    void xorHV(vBool32 *a, vBool32 *b, int size)
    {
        // Result is in place in a
        for (int i = 0; i < size; i++)
        {
            a[i] = a[i] ^ b[i];
        }
    }

    vBool32 *copy_query(vBool32 *query, int size)
    {
        // Allocate memory for the new array
        vBool32 *copy = new vBool32[size];

        // Copy the contents of the original array into the new array
        std::memcpy(copy, query, sizeof(vBool32) * size);

        return copy;
    }
    void printProfileVectors(vBool32 *profileVectors, int size)
    {
        for (int i = 0; i < 10; i++)
        {
            printf("Profile vector %d: ", i);
            printHV(profileVectors + i * size, size);
        }
    }
    int similarity(vBool32 *a, vBool32 *b, int size)
    {
        xorHV(a, b, size);
        // Result is in place in a
        // return sum of 1 in a
        int sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += sumWord(a[i][0]);
            sum += sumWord(a[i][1]);
            sum += sumWord(a[i][2]);
            sum += sumWord(a[i][3]);
        }
        return sum;
    }

    int infer(vBool32 *query, vBool32 **profileVectors, int size)
    {
        // return 0;
        int number_of_cells = (int)ceil(size / 4.0);
        // Initialize the result vector with all values set to 0
        int min_index = 0;
        uint min_value = 4294967295; // max value

        // Compute the similarity between the query and each profile vector

        for (int i = 0; i < 10; i++)
        {
            vBool32 *query_copy = copy_query(query, number_of_cells);
            int sim = similarity(query_copy, profileVectors[i], number_of_cells);
            // return 0;
            if (sim < min_value)
            {
                min_value = sim;
                min_index = i;
            }
        }

        // Return the index of the profile vector with the highest similarity
        return min_index;
    }

    vBool32 *createHV(uint32 *np, int size)
    {
        int number_of_cells = (int)ceil(size / 4.0);
        vBool32 *a = new vBool32[number_of_cells];
        for (int i = 0; i < size; i += 4)
        {
            a[i / 4] = vld1q_u32(np + i);
        }
        return a;
    }

    vBool32 *createProfileVector(vBool32 **encoderVectors, int number_of_vectors, int size)
    {

        int number_of_cells = (int)ceil(size / 4.0);
        vBool32 *profileVector = new vBool32[number_of_cells];

#if DEBUG
        std::cout
            << "Create a profile vector with " << number_of_vectors << " vectors"
            << ", each with size " << size << "." << std::endl;
        std::cout << "Number of cells: " << number_of_cells << std::endl;

#endif

        for (int i = 0; i < number_of_cells; i++)
        {

            for (int j = 0; j < 4; j++)
            {

                int *sum = new int[32]();

                for (int k = 0; k < number_of_vectors; k++)
                {
                    std::bitset<32>
                        b(encoderVectors[k][i][j]);
                    for (int l = 0; l < 32; l++)
                    {
                        sum[l] += b[l];
                    }
                }
                std::string s;
                for (int l = 31; l >= 0; l--)
                {

                    if (sum[l] > number_of_vectors / 2)
                    {
                        s += "1";
                    }
                    else
                    {
                        s += "0";
                    }
                }
                delete[] sum; // Deallocate memory
                profileVector[i][j] = std::bitset<32>(s).to_ulong();
            }
        }

        return profileVector;
    }
}

int main()
{
    // vBool32 a = {0, 0, 0, 1};
    // vBool32 b = {0, 0, 1, 1};
    // printWord(a);
    // printWord(b);
    // printWord(a ^ b);
    // printf("Hello World!");
    // const int size = 25; //
    printf("Hello HDC!\n");
    vBool32 a[1] = {{0, 0, 0, 15}};
    vBool32 b[1] = {{0, 0, 0, 0}};
    vBool32 c[1] = {{0, 0, 0, 10}};
    printf("\nA:\n");
    printHV(a, 1);
    printf("\nB:\n");
    printHV(b, 1);
    printf("\nC:\n");
    printHV(c, 1);
    printf("\nA XOR B:\n");
    xorHV(a, b, 1);
    printHV(a, 1);
    printf("\nA XOR C:\n");
    xorHV(a, c, 1);
    printHV(a, 1);
    return 0;
}