#ifndef HAVE_READLINE_HPP
#define HAVE_READLINE_HPP
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32

extern "C" {
ssize_t getline(char **line, size_t *len, FILE *fp)
{
    int c = 0;
    size_t num_bytes = 0;

    if (*line == NULL && *len <= 2) {
        *line = (char*)malloc(128);
        *len = 128;
    }
    if(feof(fp) || ferror(fp))
        return -1;

    while(true) {
        c = fgetc(fp);
        if (c == EOF)
            break;

        if ((num_bytes+2) > *len) {
            *len = *len + 128;
            *line = (char*)realloc(*line, *len);
            if (*line == NULL) {
                return -1;
            }
        }
        (*line)[num_bytes] = c;
        num_bytes++;

        if (c == '\n') // TODO how does this handle non-binary mode?
            break;
    }
    if (num_bytes == 0) {
        return -1;
    }
    (*line)[num_bytes] = '\0';
    return num_bytes;
}
}// extern C
#endif

#endif // HAVE_READLINE_HPP
