/*
 * SerialIO.h
 *
 *  Created on: Oct 3, 2017
 *      Author: jowuethrich
 */

#ifndef SERIALIO_H_
#define SERIALIO_H_

#include <stdint.h>

#define LINEBREAK ((char)'\n')
#define LINEBREAK_STR "\n"

typedef struct
{
	char (*read)();
	void (*write)(char byte);

	char lastRead;
	int lastValid;
} SerialIO;

char getChar(SerialIO* io);
void pushbackChar(SerialIO* io);


void printStr(SerialIO* io, char *str);


void printHexU12(SerialIO* io, uint16_t value);
void printHexU32(SerialIO* io, uint32_t value);
void printDecU32(SerialIO* io, uint32_t value);


uint32_t readHexU32(SerialIO* io);


uint8_t readUntil(SerialIO* io, char limit, char* buf, uint8_t bufLength);

void consumeLine(SerialIO* io);


#endif /* SERIALIO_H_ */
