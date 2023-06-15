#include <fstream>
#include <string>  // za to_string
#include <cstring> // za strtok
#include <omp.h>

void tokenize(char * lines[], int line_count)
{
    #pragma omp parallel
    {
        auto my_rank = omp_get_thread_num();
        
        #pragma omp for schedule(static,1)
        for(int i=0; i<line_count; ++i)
        {
            std::ofstream out("thread_" + std::to_string(i));
            out << "Line[" << i << "] = " << lines[i] 
                << " (thread = " << my_rank << ")\n";

            int j=0;
            char *ptr;
            //char * token = std::strtok(lines[i], " ,.!?\t\n");
            char * token = strtok_r(lines[i], " ,.!?\t\n", &ptr);
            while(token != nullptr){
                out << "Thread " << my_rank 
                    << ", token[" << j << "] = " << token << "\n";
                //token = std::strtok(nullptr,  " ,.!?\t\n");
                token = strtok_r(nullptr,  " ,.!?\t\n", &ptr);
                ++j;
            }
            out.close();
        }
    } // end parallel
}

int main() {

    std::string lines[4];
    lines[0] = "prva  li ni  ja m a l o d u ž i s t r i n g a a a a a a a a a a a a a a a";
    lines[1] = "kiša pada tra la la la a a a a a a a aa a a a a  a a";
    lines[2] = "trava raste w t f w t f t w d t  g g g g g g g g g g";
    lines[3] = "g.o.r.a   z.e.l.e.n.a fakof";

    char* ptrs[4];
    ptrs[0] = lines[0].data();
    ptrs[1] = lines[1].data();
    ptrs[2] = lines[2].data();
    ptrs[3] = lines[3].data();
    tokenize(ptrs, 4);

    return 0;
}