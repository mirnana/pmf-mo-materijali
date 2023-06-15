#include <iostream>
#include <string>
#include <omp.h>

int fun_call_cnt = 0;

struct Node {
    int value;
    Node* next;
};

bool do_work(int n) {
    const int MAX_ERRS = 10;
    bool retval = false;    // ovo je potrebno jer se u critical sekciji ne smije pozvati return
    #pragma omp critical
    {
        fun_call_cnt++;
        
        if(fun_call_cnt <= MAX_ERRS)
            retval = true;
    }

    return retval;
}

int main(int argc, char* argv[]) {

    Node * head = nullptr;
    Node * last = nullptr;

    int N = 20;
    if(argc > 1) 
        N = std::stoi(argv[1]);

    for(int i = 0; i < N; i++) {
        Node * ptr = new Node;
        if(i == 0) 
            head = ptr;
        else    
            last->next = ptr;

        ptr->value = i;
        ptr->next = nullptr;
        last = ptr;
    }

    Node* current = head;
    int ntask_failed = 0;

    double begin = omp_get_wtime();
    #pragma omp parallel firstprivate(current) shared(ntask_failed)
    {
        #pragma omp single nowait
        {
            while(current != nullptr) {

                #pragma omp task
                {
                    bool ret_val = do_work(current->value);
                    
                    
                    if(!ret_val) {
                        #pragma omp atomic 
                        ntask_failed++;
                    }
                }

                current  = current->next;   // ovo nije u tasku jer je zajednicka varijabla pa ju smije hendlati samo jedna dretva!
            }
        }   // end single
    }   // end parallel

    double end = omp_get_wtime();

    std::cout   << "fun_call_cnt = " << fun_call_cnt 
                << ", ntask_failed = " << ntask_failed 
                << ", time = " << end - begin << "\n";

    return 0;
}