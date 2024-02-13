# Topic Modeling on Dear Abby questions using LDA and NMF algorithms

## Goal:
I used unsupervised machine learning to explore underlying topics in 20,000 Dear Abby questions from 1985-2017.  This involved cleaning and preparing the data using natural language processing techniques like tokenization and lemmatization.  Then I used topic modeling algorithms including Latent Dirichlet Allocation (LDA) and Non-negative matrix factorization (NMF) to generate underlying topics from the data. 

## Data preparation
  - Data cleaning including expanding contractions, and removing symbols, puntuation, numbers, and some content-specific phrases such as copyright or booklet information
  - Tokenized, lemmatized, the data
  - Removed stopwords from the data including some content-specific generic words such as "feel", "say", "go", "dear", "abby"
  - Checked frequency words 
  - Created a dictionary and a term-document frequency corpus
  - Coherence scores indicated ideal number of topics could be 1, 2, 4, 5, or 21

Example of cleaned data:
| Original | Ready for analysis |
| ----------- | ------------------------ |
| i am newly married to a woman i'll call edith. it's the second time around for both of us. now for the problem: when we make love, edith makes me wear an undershirt. why? because i have "gretchen" tattooed on my chest above my heart. (gretchen was my first wife.) edith knew i had gretchen on my chest when she married me, but it didn't bother her. now, it's either cover up gretchen or no lovemaking. i am not used to wearing anything to bed, but unless i want to go right to sleep i have to wear an undershirt. is there some way to remove a tattoo? i've had it since i was 20, and now i'm 41. henry in elizabeth, n.j. | newly marry woman edith second time around problem love edith wear undershirt gretchen tattoo chest heart gretchen wife edith gretchen chest marry bother either cover gretchen lovemaking wear anything bed unless right sleep wear undershirt remove tattoo since henry elizabeth nj |

## Results
![wordcloud](https://github.com/KaySRubio/dear-abby-topic-modeling/assets/78124907/104ae6fd-dc5b-446e-9bc5-893a7ab410b7)

### Topic Modeling with NMF algorithm
I used Inverse document frequency (TF-IDF) Vectorization to create a document-term matrix for each unique single word (unigram) and pair of words (bigram) as input to the NMF algorithm.

Non-Negative Matrix Factorization (NMF) assumes each question belongs to only 1 topic.  Due to this assumption, the results seemed less repetitive, and more interesting compared to LDA in this project.  I also appreciated having the algorithm to pull out more topics (21) rather than fewer (4), so I'll highlight this model first.

#### NMF with 21 topics 
| Topic | Human summary | Topic Words | Example question that fell under this topic | Percent of questions in this topic |
| ------- | ---------------------- | --------------------------- | --------------------------- | --------------------------- |
| 1 | event planning | gift send card thank receive note birthday christmas buy wed money shower present check return mail invitation holiday party | like a lot of other people, the only time i write to some of my friends is during the holidays. my problem is that my husband and i are in the process of getting a divorce. we have been married for a number of years and have no children (which makes it easier), but it's still hard. how do i tell our friends? would it be proper to give them the news on a christmas card while wishing them a happy holiday? or should i just send a normal holiday card like all is well and write a separate note later? our divorce will not be final until february. how to tell | 3.2% |
| 2 | family and death| mother father mom law daughter dad day die help daughters away brother pass live sister child hurt never | my mother-in-law passed away two years ago from lung cancer. my father-in-law hasn't taken it well. this year at christmas he fabricated a letter and gifts "from her" for the grandkids, as if she had written the letter and bought the gifts before she passed away. he did it without my knowledge. i am angry and upset that i was made part of this lie without my consent. i refuse to lie to my daughter about this and plan to throw the letter away. my daughter is 6 and doesn't seem to understand. my husband doesn't think it's that big a deal and doesn't know what he can do about it. i loved my mother-in-law, but i'm tired of dealing with this. this is not the first strange thing my father-in-law has done. i feel like i get no support from my husband, who won't ever say anything to his dad. am i right in how i feel? -- don't want to lie in ohio | 3.2% | 
| 3 | marriage and divorce over time | years ago marry divorce marriage live five months move together father four never old since life die | six years ago, i did something stupid. my husband and i had two children and our marriage was rather shaky, so i let myself get talked into having my tubes tied. four years later, we divorced. seven months ago, i remarried. guess what? we want children. i was told that my surgery can be reversed - for $12,000! paid in full - up-front! this type of surgery is not covered by our insurance company. abby, i barely earn that in a year. do you know how long it will take us to save $12,000? i am 35 years old and my biological clock is running out. i know you can't help me, but maybe this letter will help someone else. women, please do not have your tubes tied if there is even the slightest chance that you will regret it later. all tied up | 4.6% |
| 4 | worry about daughters and sons | old year daughter girl woman age son boy mom man years worry problem girls little live | my 6-year-old cousin wanted to make a lemonade stand, so my sister and i helped her, but she got discouraged because nobody would buy any. she was so angry she started yelling, then she crossed the line and dropped the f-word. my sister and i were shocked that a 6-year-old would know that word. she said her classmate told it to her. (they're in kindergarten.) we told our parents, but we're not sure if we should tell her mother because she might think my sister and i taught it to her. should we tell her mother or let it slide hoping she will forget the word and move on? -- not sure in san diego | 2.5% |
| 5 | husbands, marriage, affairs | husband law marry marriage children years laws love problem affair mother ex hurt refuse picture upset ago woman | my daughter has been divorced less than a year and is dating again. (she's the one who left the marriage.) however, she keeps many pictures of her ex-husband on her facebook page. she says he was a big part of her life, and she refuses to take them down. she thinks if a guy can't accept it, then he isn't the right guy. do you agree that she's sending the wrong message? -- take the photos down | 4.9% |
| 6 | homes and visits, such as with neighbors | home house live leave visit time dinner stay neighbor move eat room night door invite day sit sleep bring food | how do you politely tell friends and relatives who are guests in your home that your computer and tv are off-limits? once they take control of the remote or the computer, they seem to go crazy and change all the settings to their preferences and never put the settings back when they leave! what can i do, abby? -- frustrated in lewis run, pa. | 5.3% | 
| 7 | weddings | wed marry bride plan invite attend fiance reception groom invitation ceremony party guests church pay couple gift brides shower | our son, 33, is getting married for the third time. his fiance has never been married and is planning a large church wedding. our son's first wedding was a traditional church ceremony with all the trimmings, showers, gifts, etc. the second time, he married a young woman who had been married before, so it was much smaller, but most of our friends and relatives sent gifts anyway. now i feel awkward sending wedding invitations to our friends and relatives a third time. i'm also afraid the bride-to-be will wonder why there will not be a wedding shower from my son's friends and relatives. how do people feel about this? we honestly don't know what to do, but we've been through all this twice already. enough is enough | 3.3% |
| 8 | raising children | children parent kid child age grandchildren dad raise young grow adult teach adopt live adults childrens abuse mom older | in a recent letter someone objected to keeping children out of school for appointments. you supported this view and requested that all physicians and dentists make their appointments with schoolchildren after school. how? i am a pedodontist and my practice is exclusively children and adolescents. if i were to follow your recommendation, i would have to schedule all my patients between 3 and 5 p.m. daily. this is impossible. please print this. perhaps some people will try to be more understanding. my poor secretary takes an awful beating from irate parents. frustrated dentist | 4.9% |
| 9 | friend issues | friend friends best friendship party close good invite hurt recently mine really guy always time lose | i am having an argument with a friend who considers himself an authority on everything. the question: who said, "the only thing we have to fear is fear itself"? i say, franklin roosevelt said it. my friend says the duke of wellington said it. who is right? big fight in little rock | 5.1% |
| 10 | schools, students, graduation | school high college class graduate grade teacher students senior girl student attend guy graduation girls parent junior friends teachers |in my opinion, we need a national slogan that reminds us to care about everybody and everything. of course, the golden rule represents this. it should be taught to children in the schools. at the beginning of the first class, its meaning could be explained. at the end of the day, teachers could remind the students, "don't forget the golden rule." what do you think, abby? -- nandor lazar, norfolk, va. | 3.2% |
| 11 | love, relationships, boyfriends, heart break | love relationship together life time boyfriend never cannot guy problem hurt really months heart break meet man ever always marry | i love my boyfriend, "joe," with all my heart; however, we have a communication problem. sometimes i feel he is dodging me or doesn't want to talk to me. joe thinks our conversations always lead to an argument, so he tries to avoid talking. joe recently moved six hours away, making it even harder to talk. i understand he may be excited about living in a new town, but i feel i deserve a little more respect than i'm getting. i'd like to talk to joe about this, but every time i call him he ignores my questions and practically hangs up on me. abby, how can i improve our communication? -- alone by the telephone | 3.1% |
| 12 | pregnancy and having babies, adoption | baby child pregnant shower sit girl birth bear months expect boy pregnancy new sitter daughter adoption adopt little grandchild |a friend of mine is expecting triplets. must everyone who attends her baby shower give her three presents, or is one gift acceptable? over budget | 2.7% |
| 13 | divorce and affairs | wife ex divorce marry marriage wifes years affair children brother man daughters recently daughter woman second picture | after 27 years of marriage, my wife told me she is attracted to other women. to my knowledge, she has acted on this only once. every day i wonder where our relationship stands. one day she can't see herself without me; the next, she says we should divorce. i don't know if i should end this or wait to see where it goes. i will need counseling if we divorce, but currently i can't afford it. yes, i love her, but what matters most to me is that she is happy. i don't have anyone else to talk to about this. any suggestions? -- mr. d. in California | 3.3% |
| 14 | work, money, offices, bosses, college | work job pay money help time bill co office workers day cannot boss buy college spend full good save | i'm a mechanic with a problem i've never seen in your column. please help me before i go nuts. have you ever worked with a whistler? at 10 minutes to 8 in the morning, i can hear whistling as he is coming into the shop. and he whistles for eight hours continuously! no tune--just whistling. i don't know whether to cry, throw something at him, choke him or what. one day he was out sick, and i thought i had died and gone to heaven! i finally told the boss. he said if i didn't like it, i could quit. (the whistler is his brother-in-law.) i need this job. what do you suggest? going nuts in dunkirk, n.y. | 5.9% |
| 15 | engagement stuff, like rings, dress, jewelry | wear ring dress engagement buy hair clothe phone white abbys color finger jewelry black short new beautiful shirt hat | my boyfriend surprised me with a diamond engagement ring for christmas. it wasn't cheap by any means, but i hated it. now the problem. i went to the jewelry store it came from and exchanged it for the kind i wanted-a solitaire. i've been married before and i hated my first engagement ring, so this time i wanted one i really liked, so i got a solitaire. i love my boyfriend with all my heart, and i wouldn't hurt his feelings for the world, but i'm afraid i did. i could see the disappointment in his eyes when i told him i had exchanged my ring for a solitaire. he admitted he felt hurt, but he never brought the subject up again. was i wrong to have exchanged the ring? i've been put down by family members. what do you think? put down in Canada | 3.2% |
| 16 | sons, fathers, grandsons | son father old sons boy law child daughter girlfriend boys dad grandson refuse young help part answer support play trouble | i need help. my son joined the army, then after he finished basic training, he took off without leave. so far, he's still running, calling me whenever he can. he called last night saying he was tired of running wants to give himself up. i need to know what the army will do to him for running away. will he have to go to prison? will they beat him? he's only 20. please answer soon because he is waiting for your answer. can't sign this | 3.3% |
| 17 | pets and neighbor's pets | dog pet cat neighbor animal sleep owner put allow bite walk train clean service bring care lover house yard front | i am 8 years old. my mom told me our neighbor's dog was old and sick, so he had to put his dog to sleep. i hate this. i know it is what is best for the dog, but i can't stop thinking about it. how can i get over this? -- henry in austin, texas | 1.2% |
| 18 | brothers, sisters, holidays | family sister brother members law sisters member holiday close gather brothers include invite hurt attend friends | i would like you to settle a major family dispute once and for all. i am a 20-year-old college student who comes from a working-class family and grew up in a blue-collar community. the dispute is this: is there a distinct difference between a "profession" and a "trade" (job)? my family says a person with a trade is a professional in that area, so there can be no distinction between the two. i disagree. i say lawyers, doctors, teachers, etc. are "professionals" and roofers, auto mechanics, construction workers, etc. are "trade people." is there a difference? profession or trade, river rouge, mich. | 4.9% |
| 19 | sex and attraction | man women men woman sex meet marry young single age guy find interest male attractive hair female attract never | how can i give my boyfriend makeup sex if we never have an argument? -- miss bliss in Indiana | 5.3% |
| 20 | logistics of the column | people letter question read answer hear person column may thank readers word print doctor others smoke find sign remember time | you've heard from the jacks and the chucks, and then you said, "now let's hear from the johns." haven't you heard? johnny can't read. johnny can't write. johnny can't seem to do anything. maybe that's why they say, "let george do it." john can in oregon every teen-ager should know the truth about drugs, sex and how to be happy. for abby's booklet, also available in spanish, send your name and address clearly printed with a check or money order for $2.50 (this includes postage) to: abby, teen booklet, p.o. box 38923, hollywood, calif. 90038. | 4.9% | 
| 21 | logistics of the column | envelope stamp angeles self address los po box calif personal enclose reply | i'm surprised you didn't recommend silicone implants to "flat-chested." i spent 39 years hating my body, then i decided to have breast- augmentation surgery. i'm only sorry i waited so long. looking great problems? what's bugging you? unload on abby, p.o. box 38923, hollywood, calif. 90038. for a personal reply, please enclose a stamped, addressed envelope. | 22.4% |
* The last 2 topics are more related to the logistics of the Dear Abby column itself.  I removed some copyright phrases and any content after the word "booklet" but there project could benefit from additional cleaning.

#### NMF with 4 topics 
| Topic | Human summary | Words from algorithm |
| ------- | ---------------------- | --------------------------- |
| 1 | work, friends | people work time home friends day friend letter help person hear good read house job husband find eat leave cannot |
| 2 | weddings and other parties | wed gift send thank card receive bride invite attend party marry family shower note invitation plan birthday money |
| 3 | all family | mother old year daughter son parent father mom child children baby girl dad family law school live sister boy |
| 4 | marriage & divorce | love years marry husband wife man marriage divorce relationship together ago woman life time never meet months children live |

### Topic Modeling with LDA algorithm
In contrast with NMF, LDA algorithm assumes all questions share topics but have different weightings of those topics. I found that this algorithm tended to identify topics focused heavily on time, husbands, and mothers, with some additional more minor themes.  

#### LDA algorithm with 2 topics
Topic 1:
0.009 * time + 0.007 * years + 0.007 * husband + 0.007 * mother + 0.007 * year + 0.006 * marry + 0.006 * live + 0.006 * old + 0.006 * work + 0.005 * family

Topic 2:
0.011 * years + 0.009 * husband + 0.009 * time + 0.007 * love + 0.006 * year + 0.006 * people + 0.006 * family + 0.005 * old + 0.005 * friends + 0.005 * never

#### LDA algorithm with 5 topics
Topic 1:
0.012 * years + 0.011 * time + 0.010 * husband + 0.007 * year + 0.006 * love + 0.006 * wife + 0.006 * family + 0.005 * people + 0.005 * old + 0.005 * home

Topic 2:
0.009 * years + 0.008 * daughter + 0.007 * old + 0.007 * love + 0.006 * never + 0.006 * year + 0.006 * time + 0.006 * marry + 0.006 * people + 0.005 * husband),

Topic 3:
0.010 * years + 0.008 * time + 0.006 * people + 0.006 * love + 0.006 * old + 0.006 * family + 0.006 * husband + 0.006 * live + 0.006 * work + 0.005 * day

Topic 4:
0.010 * husband + 0.009 * years + 0.009 * mother + 0.009 * time + 0.008 * year + 0.008 * marry + 0.007 * live + 0.007 * children + 0.007 * family + 0.007 * son),

Topic 5:
0.010 * time + 0.008 * husband + 0.007 * never + 0.007 * year + 0.006 * love + 0.006 * cannot + 0.005 * years + 0.005 * people + 0.005 * family + 0.005 * mother

![Screenshot 2024-02-09 at 5 40 28 PM](https://github.com/KaySRubio/dear-abby-topic-modeling/assets/78124907/e63af278-09cd-4e71-a182-2bb11efd7a53)


#### LDA algorithm with 21 topics
Topic 1: 
0.010 * time + 0.010 * years + 0.008 * wife + 0.007 * husband + 0.007 * people + 0.006 * never + 0.006 * children + 0.006 * love + 0.006 * son + 0.006 * marry),

Topic 2:
0.010 * marry + 0.009 * time + 0.009 * love + 0.009 * years + 0.008 * never + 0.006 * work + 0.006 * man + 0.006 * family + 0.006 * problem + 0.006 * husband

Topic 3: 
0.014 * years + 0.011 * year + 0.008 * husband + 0.008 * live + 0.007 * mother + 0.006 * time + 0.006 * marry + 0.006 * friends + 0.005 * move + 0.005 * life

Topic 4: 
0.014 * husband + 0.008 * years + 0.008 * time + 0.006 * marry + 0.006 * year + 0.006 * parent + 0.005 * friends + 0.005 * help + 0.005 * day + 0.005 * wife),

Topic 5: 
0.010 * time + 0.007 * mother + 0.007 * home + 0.007 * husband + 0.006 * family + 0.006 * people + 0.006 * live + 0.006 * marry + 0.006 * children + 0.005 * work),

Topic 6:
0.011 * help + 0.008 * children + 0.008 * live + 0.007 * love + 0.007 * years + 0.007 * mother + 0.006 * work + 0.006 * school + 
  0.006 * old + 0.005 * time

Topic 7:
0.009 * years + 0.008 * old + 0.007 * time + 0.006 * friend + 0.005 * family + 0.005 * people + 0.005 * live + 0.005 * man + 0.005 * year + 0.005 * friends

Topic 8:
0.012 * years + 0.008 * home + 0.008 * time + 0.008 * mother + 0.007 * never + 0.007 * live + 0.006 * husband + 0.006 * marry + 0.006 * old + 0.006 * love

Topic 9:
0.011 * years + 0.009 * year + 0.009 * never + 0.009 * old + 0.008 * marry + 0.008 * people + 0.007 * husband + 0.007 * friends + 0.005 * time + 0.005 * live

Topic 10:
0.013 * mother + 0.011 * husband + 0.009 * time + 0.009 * love + 0.008 * children + 0.007 * years + 0.007 * child + 0.007 * wed + 0.006 * son + 0.006 * marry

Topic 11:
0.010 * year + 0.009 * old + 0.009 * people + 0.008 * husband + 0.008 * years + 0.008 * time + 0.007 * family + 0.006 * friends + 0.006 * daughter + 0.006 * love

Topic 12:
0.010 * work + 0.009 * husband + 0.007 * family + 0.006 * year + 0.006 * son + 0.006 * live + 0.006 * years + 0.006 * daughter + 0.006 * marry + 0.005 * time

Topic 13:
0.010 * years + 0.009 * love + 0.007 * man + 0.007 * time + 0.006 * work + 0.006 * live + 0.006 * people + 0.006 * year + 0.006 * woman + 0.006 * never

Topic 14:
0.016 * time + 0.010 * parent + 0.009 * mother + 0.008 * love + 0.008 * years + 0.007 * year + 0.006 * friends + 0.006 * good + 0.006 * mom + 0.006 * life

Topic 15:
0.010 * time + 0.010 * gift + 0.009 * years + 0.008 * mother + 0.007 * marry + 0.007 * love + 0.007 * home + 0.007 * never + 0.006 * husband + 0.006 * work

Topic 16:
0.015 * husband + 0.013 * years + 0.011 * time + 0.008 * year + 0.007 * cannot + 0.007 * family + 0.006 * mother + 0.006 * live + 0.006 * old + 0.006 * wife

Topic 17:
0.012 * years + 0.008 * time + 0.008 * family + 0.007 * love + 0.007 * home  + 0.006 * husband + 0.005 * new + 0.005 * work + 0.005 * friends + 0.005 * people

Topic 18:
0.012 * time + 0.009 * marry + 0.008 * years + 0.008 * mother + 0.008 * year + 0.007 * daughter + 0.007 * live + 0.006 * old + 
0.005 * family + 0.005 * love

Topic 20:
0.012 * husband + 0.008 * years + 0.008 * family + 0.007 * people + 0.007 * love + 0.006 * work + 0.006 * live + 0.005 * leave + 0.005 * year + 0.005 * friends

Topic 21:
0.011 * husband + 0.008 * years + 0.007 * people + 0.007 * home + 0.007 * time + 0.007 * old + 0.006 * daughter + 0.006 * year + 0.005 * family + 0.005 * children

*Note: Topic 19 was mysteriously missing from the output of this model.  If this were my favorite model, I'd do some debugging on this, but since it's not, I moved on.

![Screenshot 2024-02-09 at 5 41 32 PM](https://github.com/KaySRubio/dear-abby-topic-modeling/assets/78124907/2a2c0ce6-318f-4807-8dc9-1e29072e80fe)

## Next steps:
Some cool next directions would be to see if topic frequencies have changed over time from 1985-2017.  Also, there's always more data cleaning to do.  While I excluded some column-related phrasing and words, I discovered that some logistical info was still left in, so more rigorous data cleaning could.  I'd also like to try Latent Semantic Analysis (LSA) and other topic modeling algorithms. 

## Data Source: 
American Anxieties: Dear Abby's Questions
20,000 Questions to Dear Abby: Insights on American Anxieties
By Kelly Garrett
https://www.kaggle.com/datasets/thedevastator/american-anxieties-dear-abby-s-questions

