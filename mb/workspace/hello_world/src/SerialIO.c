/*
 * SerialIO.c
 *
 *  Created on: Oct 3, 2017
 *      Author: jowuethrich
 */

#include "SerialIO.h"


char getChar(SerialIO* io)
{
	if(!(io->lastValid))
	{
		io->lastRead = io->read();
	}

	io->lastValid = 0;
	return io->lastRead;
}
void pushbackChar(SerialIO* io)
{
	io->lastValid = 1;
}


void printStr(SerialIO* io, char *str)
{
	while(*str)
	{
		io->write(*str);
		str++;
	}
}


void printHexU12(SerialIO* io, uint16_t value)
{
	char str[4];
	str[3] = '\0';

	int ii;
	for(ii=0; ii<3; ii++)
	{
		uint16_t v = value & 0x0000000F;
		if(v < 10)
		{
			str[2 - ii] = '0' + v;
		} else
		{
			str[2 - ii] = 'A' + v - 10;
		}
		value >>= 4;
	}

	printStr(io, str);
}

void printHexU32(SerialIO* io, uint32_t value)
{
	char str[9];
	str[8] = '\0';

	int ii;
	for(ii=0; ii<8; ii++)
	{
		uint32_t v = value & 0x0000000F;
		if(v < 10)
		{
			str[7 - ii] = '0' + v;
		} else
		{
			str[7 - ii] = 'A' + v - 10;
		}
		value >>= 4;
	}

	printStr(io, str);
}

void printDecU32(SerialIO* io, uint32_t value)
{
	char str[10];
	str[9] = '\0';

	int ii;
	for(ii=0; ii<9; ii++)
	{
		str[8 - ii] = '0' + (value%10);
		value /= 10;
	}

	printStr(io, str);
}

uint32_t readHexU32(SerialIO* io)
{
	uint32_t value = 0;

	while(1)
	{
		uint8_t c_val = 0;
		char c = getChar(io);

		if(c >= 'A' && c <= 'F')
		{
			c_val = (c - 'A' + 10);
		} else if(c >= 'a' && c <= 'f')
		{
			c_val = (c - 'a' + 10);
		} else if(c >= '0' && c <= '9')
		{
			c_val = (c - '0');
		} else
		{
			pushbackChar(io);
			break;
		}

		value <<= 4;
		value += c_val;
	}

	// TODO

	return value;
}

uint8_t readUntil(SerialIO* io, char limit, char* buf, uint8_t bufLength)
{
	uint8_t fill = 0;
	char c = getChar(io);
	while(fill < bufLength-1 && c != limit && c != LINEBREAK)
	{
		buf[fill] = c;
		fill++;
		c = getChar(io);
	}
	buf[fill] = '\0';

	if(c == LINEBREAK)
	{
		// Do not consume any LINEBREAKS
		pushbackChar(io);
	}

	return fill;
}

void consumeLine(SerialIO* io)
{
	while(getChar(io) != LINEBREAK) ;
}

