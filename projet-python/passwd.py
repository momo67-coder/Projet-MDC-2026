#!/usr/bin/python3

import random
import string
import hashlib


class Password:
    """
    Classe de generation de mot de passe.
    """

    __lstChar = ['$', '#', 'r', '@', 'U', '4', 'w']
    __words_leet = ['eleet', 'warlords', 'magik', 'exploit', 'hacker', 'leetspeak', 'madskills', 'theforce', 'fortytwo',
                    'advanced', 'geek', 'bingo', 'ultra']

    def __init__(self, level):
        """
        Creation d'un mot de passe

        Parameters
        ----------
        level: int
            Niveau du mot de passe à générer
        """
        self.__level = level
        self._o_ = lambda x: x^0x1e53b09
        self.__hashcode = hashlib.sha256()
        if level == 1:
            self.gen_char(3)
        elif level == 2:
            self.gen_int(6)
        elif level == 3:
            self.gen_all(3)
        elif level == 4:
            self.gen_lst(8)
        elif level == 1337:
            self.gen_leet()
        elif level == 31337:
            self.gen_eleet()
        else:
            self.gen_char(1)

    def o_O(self, x):
        keys = {0: 35520119, 1: 63244890, 2: 22550716, 3: 56777434,
                4: 49763350, 1337: 57136103, 31337: 21969110}
        check = hashlib.sha256(''.join(x).encode('ascii'))

        if check.hexdigest() != self.__hashcode.hexdigest():
            return -1
        return keys[self.__level]
        
    def gen_char(self, size):
        """
        Creation d'un mot de passe compose de lettres minuscules
        """
        passwd = [random.choice(string.ascii_lowercase)
                  for _ in range(size)]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def gen_int(self, size):
        """
        Creation d'un mot de passe compose de chiffres
        """
        passwd = [random.choice(string.digits) for _ in range(size)]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def gen_all(self, size):
        """
        Creation d'un mot de passe compose de caracteres entre ' ' et '~'
        """
        f = ord(' ')
        l = ord('~')
        passwd = [chr(random.randint(f, l)) for _ in range(size)]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def gen_lst(self, size):
        """
        Creation d'un mot de passe compose de caracteres selectionnes parmi
        une liste
        """
        s = len(Password.__lstChar)
        passwd = [Password.__lstChar[random.randint(0, s - 1)]
                  for _ in range(size)]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def __init_leet(self):
        leet = dict()
        for x in string.ascii_lowercase:
            leet[x] = [x.upper()]
        leet['a'].extend(['4', '@'])
        leet['b'].append('8')
        leet['c'].append('(')
        leet['e'].append('3')
        leet['g'].extend(['6', '&'])
        leet['h'].append('#')
        leet['i'].extend(['!', '|'])
        leet['l'].append('1')
        leet['o'].append('0')
        leet['s'].extend(['5', 'Z', '$'])
        leet['t'].append('7')
        leet['z'].append('2')
        return leet

    def __init_eleet(self):
        leet = self.__init_leet()
        leet['b'].append('|3')
        leet['e'].append('[-')
        leet['f'].append('|=')
        leet['h'].append('|-|')
        leet['k'].append('|<')
        leet['l'].append('|_')
        leet['m'].append('|\/|')
        leet['m'].append('|^^|')
        leet['n'].append('|\|')
        leet['p'].append('|*')
        leet['v'].append('\/')
        leet['w'].append('\/\/')
        leet['x'].append('><')
        return leet

    def gen_leet(self):
        word = random.choice(Password.__words_leet)
        leet = self.__init_leet()
        passwd = [random.choice(leet[x] + [x]) for x in word]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def gen_eleet(self):
        word = random.choice(Password.__words_leet)
        leet = self.__init_eleet()
        eleet_ok = False
        while not eleet_ok:
            tmp = ''.join([random.choice(leet[x] + [x]) for x in word])
            eleet_ok = (len(tmp) != len(word))
        passwd = [x for x in tmp]
        self.__hashcode.update(''.join(passwd).encode('ascii'))

    def check(self, passw):
        f = lambda x: -1 if x==-1 else self._o_(x)
        return f(self.o_O(passw))