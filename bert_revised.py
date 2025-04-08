import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import os
import json
from tqdm import tqdm
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import our data loading and preprocessing modules
from data_loader import (
    load_data_from_csv, 
    load_data_from_txt,
    create_pronoun_mappings,
    prepare_dataset,
    prepare_train_val_test_split
)

class PronounClassifierTrainer:
    """Class to handle training and evaluation of the pronoun classifier model."""
    
    def __init__(self, model, device: torch.device, output_dir: str = "./pronoun_classifier_model"):
        """
        Initialize the trainer.
        
        Args:
            model: BERT model for sequence classification
            device: Device to train on (cuda/cpu)
            output_dir: Directory to save the model and results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store training stats
        self.training_stats = []
        self.best_val_loss = float('inf')
        
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        early_stopping_patience: int = 2,
        max_grad_norm: float = 1.0
    ) -> Tuple[Dict, List[Dict]]:
        """
        Train the BERT model for pronoun classification.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            early_stopping_patience: Number of epochs to wait before early stopping
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple containing the best model state dict and training statistics
        """
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early stopping
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 40)
            
            # Training phase
            avg_train_loss, train_report = self._run_epoch(
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                is_training=True,
                max_grad_norm=max_grad_norm,
                desc="Training"
            )
            
            # Validation phase
            avg_val_loss, val_report = self._run_epoch(
                dataloader=val_dataloader,
                desc="Validation"
            )
            
            # Print epoch statistics
            print(f"\nAverage training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print("\nTraining classification report:")
            print(classification_report(train_report['labels'], train_report['preds']))
            print("\nValidation classification report:")
            print(classification_report(val_report['labels'], val_report['preds']))
            
            # Store stats for this epoch
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_report': classification_report(train_report['labels'], train_report['preds'], output_dict=True),
                'val_report': classification_report(val_report['labels'], val_report['preds'], output_dict=True)
            }
            self.training_stats.append(epoch_stats)
            
            # Check if this is the best model so far
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                self._save_model("best_model")
                print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Save training stats and final model
        self._save_training_stats()
        self._save_model("final_model")
        
        return self.model, self.training_stats
    
    def _run_epoch(
        self,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        is_training: bool = False,
        max_grad_norm: float = 1.0,
        desc: str = ""
    ) -> Tuple[float, Dict]:
        """
        Run a single epoch of training or validation.
        
        Args:
            dataloader: DataLoader for the current phase
            optimizer: Optimizer (only for training)
            scheduler: Learning rate scheduler (only for training)
            is_training: Whether this is a training epoch
            max_grad_norm: Maximum gradient norm for clipping
            desc: Description for progress bar
            
        Returns:
            Tuple containing average loss and report dictionary with predictions and labels
        """
        if is_training:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc=desc)
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Clear gradients if training
            if is_training:
                self.model.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(is_training):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
            # Backward pass and optimization if training
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate predictions
            preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()
            label_ids = labels.cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(preds)
            all_labels.extend(label_ids)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, {'preds': all_preds, 'labels': all_labels}
    
    def evaluate(self, dataloader: DataLoader, desc: str = "Evaluation") -> Dict:
        """
        Evaluate the model on a given dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            desc: Description for progress bar
            
        Returns:
            Dictionary containing evaluation results
        """
        avg_loss, report = self._run_epoch(dataloader, desc=desc)
        
        results = {
            'loss': avg_loss,
            'report': classification_report(report['labels'], report['preds'], output_dict=True),
            'classification_report': classification_report(report['labels'], report['preds'])
        }
        
        return results
    
    def _save_model(self, model_name: str):
        """Save the model to the output directory."""
        model_path = os.path.join(self.output_dir, model_name)
        self.model.save_pretrained(model_path)
    
    def _save_training_stats(self):
        """Save training statistics to a JSON file."""
        stats_path = os.path.join(self.output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

def predict_pronouns(
    texts: List[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    idx_to_pronoun: Dict[int, str],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512
) -> List[str]:
    """
    Predict the primary pronoun for each text.
    
    Args:
        texts: List of texts to predict pronouns for
        model: Trained BERT model
        tokenizer: BERT tokenizer
        idx_to_pronoun: Mapping from indices to pronouns
        device: Device to use for inference
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        List of predicted pronouns for each text
    """
    # Prepare dataset for inference
    dataset = prepare_dataset(texts, tokenizer, is_training=False, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Set model to evaluation mode
    model.eval()
    all_predictions = []
    
    # Progress bar for inference
    progress_bar = tqdm(dataloader, desc="Prediction")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        
        # No gradient calculation needed for inference
        with torch.no_grad():
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            
            # Calculate predictions
            preds = torch.argmax(logits, dim=1).flatten().cpu().numpy()
            all_predictions.extend(preds)
    
    # Convert prediction indices to pronoun names
    pronoun_predictions = [idx_to_pronoun[idx] for idx in all_predictions]
    
    return pronoun_predictions

def save_pronoun_mappings(
    pronoun_to_idx: Dict[str, int],
    idx_to_pronoun: Dict[int, str],
    output_dir: str
) -> None:
    """
    Save pronoun mappings to a JSON file.
    
    Args:
        pronoun_to_idx: Mapping from pronouns to indices
        idx_to_pronoun: Mapping from indices to pronouns
        output_dir: Directory to save the mappings
    """
    os.makedirs(output_dir, exist_ok=True)
    mappings_path = os.path.join(output_dir, 'pronoun_mappings.json')
    with open(mappings_path, 'w') as f:
        json.dump({
            'pronoun_to_idx': pronoun_to_idx,
            'idx_to_pronoun': idx_to_pronoun
        }, f, indent=2)

def main(
    data_path: Optional[str] = None,
    model_name: str = 'bert-base-uncased',
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    output_dir: str = './pronoun_classifier_model',
    early_stopping_patience: int = 2
) -> None:
    """
    Main function to train and evaluate the pronoun classifier.
    
    Args:
        data_path: Path to data file (CSV or TXT)
        model_name: Pre-trained BERT model name
        batch_size: Batch size for training and inference
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        max_length: Maximum sequence length
        output_dir: Directory to save the model and results
        early_stopping_patience: Number of epochs to wait before early stopping
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Sample data (replace with your actual data loading)
    sample_texts = [
        """This is what it takes to get into Harvard University...""",
        """This is what it takes to get into Harvard University. Here's a college application of a girl who did and into Harvard. She had a 4.1 GPA, a 1560 SAT score, and a 36, a perfect score on the AC. Comes from a high income family, her dad went to Harvard and her mom went to UCLA. He intense the major in business, his valedictorian, and she only had two extracurriculars in school. She did varsity track and varsity cross country. On top of that, she also had a business that made $30,000 in profit last year. So something that she started, her sophomore year. And as for her college results, well, she was accepted to Georgetown. Ah, I'm sorry. She actually did not get into Harvard. She was accepted to Clemson, USC, and UNC. She was rejected from Yale and UPEN. And she was rejected from MIT and Duke as well. So this is not what Harvard wants. My dad. But still a very solid application. And it's not easy making a business that has $30,000 in profit every year. So good job to her. Guessing she would probably be at either Georgetown or UNC or USC.""",
            "Here are the universities you get into with a 3.8 GPA in a 1250 SAT. You get accepted to Florida State, Florida International, University of Tampa, University of Alabama, University of Baylor, but not Texas, Christian University, North University of Florida. As for the rest of the students, college application, while they took 4 APs, was in the top 20% of their school in terms of GPA, is White Inn, Hispanic, Third Gen, comes from a high income family, was a member of Student Council, President of Photography Club, Key Club Member, Opposing Member, and Creative of a Service Project, Benefit Essential American, and ICU Facilities. He sent 55 boxes of diapers and 63 packs of wipes. No bed!",
            "Here's a college application for students who got into Harvard as an athlete. 96 out of 100 GPA. Player of the year in Europe. Talent of the year in Slovakia. Junior of the year 2023. Junior Roland Garros. Quarter Finalists in singles and semi-finals in doubles. Ranked UAT in junior in the world. Ranked in professional men's tennis ranking. Participated in the junior US Open. Wimbledon and Roland Garros two times. Number one ranked U16 in Europe for two years straight. That's it. He was accepted and committed to Harvard. He's majoring in psychology and government. So the secret, get really good at a sport. And I was just thought you.",
            "Here are the stats of a student who got into Harvard. 4.48 weighted GPA, 1540 SAT. He found out a nonprofit that raised a lot of money for cancer in three years. The medical research on cancer with a mentor from Cornell that he published. In turn, that dentistry office for two summers. Volunteered at a sick church every Sunday and held lunches for those in need. Founded a fast math for kids tutoring business, coached kids recreational basketball, and picked up trash in the neighborhood with his family every month during the summer. Here's a Coca-Cola scholar-send-my-finalist, was published in the local newspaper and his CPR certified. He's an upper-metal class Indian and a first-gen college student. As mentioned before, he was accepted to Harvard, but he was weightless different from Columbia, accepted to Brown and Dartmouth, but rejected from Yale. He was accepted to UMass and Boston College as well. Columbia and Yelda Noanum, but Harvard did. An arguably Harvard may seem like the most prestigious university on this list, but hey, it's all about fit and how you look to the university. Best of luck.",
            "Just gonna say that this valetorian has a pretty average college application. Yes, she had 4.5 GPA, yes, she was valetorian, yes, she had probably two questions wrong on her SAT total. But I mean, if you look at her awards, there's really nothing crazy. Book Award, Sports Hall, Fame Student Athlete, School's Top Award for Citizenship, School's Top Award for Leadership. I mean, if you look at her clubs, she was part of the engineering program, Special Olympics Club, One Love Club, all stuff that she just did at school. I'm not saying it's a bad thing, but it's just not really anything particularly special. She only applied to what? Five schools here? Princeton, UNC, UBA, Duke, and Washington and Lee. Now, if I had to guess, I would probably say he had a tough chance at Princeton and Duke, given that they are top 20s. And you'd be very surprised as to where he actually got in. All for part 2 to see the results. You will not believe which university's this average high school student got into. The student in my previous video had a 1580 SAT and was valetorian. Now, hear me out. Please check out the previous video to see why her application was average, in my opinion. See, the thing is, if you don't have a theme with your application, it's hard to stand out. And this student didn't really have a theme. However, she's currently at Princeton University. Not only that, but she got accepted to every single school that she applied to. Her essays were probably top notch. Let me know if you want to hear what the college essay of a Princeton student sounds like. In this video, I said that this valetorian's application was average. Let me make a clarification. Remember when I apologized, sometimes your boy got to do some clickbaiting to get some views. I'm sorry, I'm sorry. Number 2, as someone who has read hundreds and hundreds of applications at this point. A lot of the applications that you see that get into the Ivy League, that get into Harvard, are insane applications. That isn't to say that the student that I reviewed in the video was average. Yes, by the average student standard, 1580 SAT and valetorian is nowhere near average. Clearly. But the thing is, most of my content is focused on top 20 admissions. So how to get into the top 20 universities in the world. Generally speaking, great stats only gets you so far. The average SAT of a Harvard applicant last year was something like a 1560. For 8,000 valetorians, I applied to Harvard every single year. But yes, I apologize. It was a little bit misleading. I didn't mean to say that this student was average from an average American student standpoint. I'm talking average on the standpoint of a student who's applying to a top 20 university. That's all.",
            "Here are the colleges you get into with a 3.4 GPA and an 1110 SAT score. This is not applied to UT Austin, Texas, and M. Baylor, UT Dallas, University of Houston, Lamar University, University of North Texas, Sam Houston State, and SFA. If you can guess what state this guy is from, I'll give you a dollar. Guess what? He was accepted to every single one of those. As for the rest of his college application, while he took two APs, Kam and Calculus AB. He's Indian, second gen comes from a middle income family. He was president of Student Council, NHS Treasurer, had placement in IT services and IT Quiz Bowl, with a tech intern with his school district. At 40 plus dual enrollment hours, 10 plus first division medals in band and was section leader. There you have it. Pretty solid application, and he's probably killing it at whatever university he decided to go to. He didn't indicate here yet.",
            "Here's a college application of a student with a 3.8 GPA and a 1370 SAT, as well as her extra curriculars, awards, and what universities she got into. Well, she was on a roll all four years, fourth place at Decaregional, seventh place at Deca States. She took AP Lang, AP Psych, AP Lit, AP Econ, and AP Art History. As for her extra curriculars, well, she played JP basketball, ninth grade, varsity, tenth, and eleventh. She played AU basketball as well. What's part of Yearbook Club had a part time job? What's part of Volunteer Club? Deca, volunteer camp counselor, and did volunteer work with Children of Cancer all four years of high school. And as for her college results, well, she was accepted to ASU, accepted to JMU, deferred from South Carolina, accepted to BT, deferred that accepted to SMU, accepted to Auburn, deferred then waitlisted from Clemson, rejected from UGA, accepted to UCF, accepted to STSU, deferred from Tennessee, rejected from UF, and accepted to FSU. And she's currently attending FSU. There you have it, folks.",
            "Here are the universities you get into with a 3.5 GPA and an 1170 SAT school. Just for some additional background, this student is Asian and White, comes from a high income family and is an only child. On top of those stats, he played varsity with a cross for two years. What's part of national honor society was also part of varsity track. Worked for a local design clothing company that made 10K revenue. He was also a tennis instructor for two years, was second-chair clarinet for school band, and was also part of the top 50 team in the country, club and lacrosse. And asked for his college results, well, here we go. Denied from college of holy cross, accepted to Bentley, accepted to a Delphi University, accepted to Connecticut College, denied from Fairfield University, denied from Fordham, accepted to Holt International, denied from Northeastern, accepted an attending Providence College, accepted to Knippy Act with an $82,000 scholarship, accepted to Rutgers, accepted to Sacred Heart with a 93K scholarship, accepted to Salad, Virginia University, accepted to St. John's with a $120,000 scholarship, accepted to Suffolk, a University of Scranton, and also accepted to the University of Massachusetts in Boston. There you have it, solid college results, solid application, and he's probably killing it at Providence.",
            "This college application is proof that you don't need insane extracurriculars to get into a top university. This student got into Stanford with her main extracurriculars being three years of varsity basketball, three years of varsity golf, two years of varsity badminton, two years of javvy badminton. She did have a nonprofit that generated 30k over four years, but nothing more specific than that. She also did have an online business that generated 3k a month, was part of speech and debate, and was student body president. So with a little bit of hard work and dedication, most people can realistically accomplish things that are similar to these extracurriculars. I will say that this student had a 4.8 weighted GPA, was ranked second out of 115 kids in her class, and had a 1590 SAT, as well as a 35 on her ACT. As for the colleges that she got into, well, Stanford is number one. She's also accepted to UCLA, Columbia, Duke, UCSD, and UCI. Very notably, she was rejected from Cornell, rejected from Yale, Harvard, MIT, Berkeley, and USC. So clearly, some schools thought that she was a great fit with them, some schools didn't. But hey, listen, it only takes one, remember?",
            "This is the most insane college application I've ever seen in my life. I'm gonna try to guess which university you got into. For starters, it's got two New York Times bestselling books featured in a national TV documentary. Arab America 20 under 20, what the hell, I didn't even know that existed. Two times national chemistry Olympiad finalist, first of all, Bra. Started a math and physics page with combined 200,000 followers. So it's got that cloud two, hold over 10,000 books generating 300 can revenue for organic content. I'm an captain of science team turned out world from research at 17 years old. I'm also the wrestling, boxing, football, and Dawson to ask Smithsonian. Alright, well, here the school is going to apply to. Obviously, when you're this good of a student, you're going to apply to every school top 20. He's definitely going to get enough for Stanford. I'm going to guess he got it Stanford. I'm going to guess rejected from Harvard though. Accepted to MIT, rejected from Yale, accepted to Princeton. I'm going to say accepted to Caltech as well. Accepted to UVA, watch you, Carnegie Mellon, NYU, rejected from Columbia. I'm going to say rejected from Northwestern and accepted to. Accepted to UMish, urbanic champagne, West Point, UC Boulder, and the Sunnis. Oh, forgot to mention, he also has a 4.2 GPA and a 1560 SAT. Ball for part two, if you want to see which schools he actually got into.",
            "Here what college you should get into with a 3.2 GPA and a 25 on your ACT. The STEM play varsity basketball for all four years, golf for three years, tennis for three years, violin for four years, and was leader of honorco, as well as the first place in a West Coast competition. She made it to state several years in a row for robotics and had 200 hours of volunteer work. She's Asian and she's from California. She was rejected from UCLA, rejected from Berkeley, accepted to Oregon State, rejected from Loyola, Marymount, accepted to Colorado Boulder, accepted to the University of San Francisco, rejected from Santa Clara, and accepted to Colorado Springs. She's currently at Oregon State and she's majoring in management and marketing.",
            "This is the college application of a student that was accepted to MIT, Harvard, Johns Hopkins, Caltech, UC Berkeley, UT Austin, and the University of Wisconsin, but was rejected from Stanford, waitlisted from Princeton, and rejected from the University of Illinois. The student had a 4.0 unwitting GPA, a 1550 on her SAT. She was valedictorian, ranked 1 out of 1300 students in her class. She was Varsity Soccer Captain, president of the chemistry club. Varsity Tracking Field was part of the Science Olympiad team. Math team, volunteer tutor, research assistant, passion project, about renewable energy production from biomass waste, and also one multiple science fair regionals and one to ISF using nanoparticle and hands drug delivery system for targeted cancer therapy. The first sentence of her essay goes, In the tapestry of my life, I am living a symphony woven from the threads of my Cuban, battalion German and Irish heritage. She's at MIT, symphony woven from the threads of my Cuban, battalion German and Irish heritage. She's at MIT.",
            "Here are what universities you get into when you are number six in the world for math. This girl placed sixth at the International Math Olympia. The most prestigious and most difficult math competition globally. Not to mention she was valedictorian ranked one out of 400 kids in her class with a 1580 SAT and 11 AP courses under her belt. She didn't have much other activities though. She did some volunteer work and was part of model UN but I'd imagine doing math for seven hours every day to practice for the International Math Olympia. It probably takes up most of your day. And as for the universities that she got into. Well, to no surprise, she got into Harvard. Oh, well, she also got into Stanford and you Chicago and Columbia and Princeton and Cornell and UC Berkeley. Tufts, Fordham, UVA, Bowden and while you stern and molesting. But very notably, she was rejected from Dartmouth, UPEN, Yale, Brown and Williams. You can't win them all but doesn't matter when you get accepted to Harvard in Princeton, which by the way, she chose Princeton because we want to go to snobby Harvard. Nothing crazy just number six in the world for math.",
            "Here are what universities, a valedictorian with fairly average extracurriculars gets into. This student had a 4.0 unweighted 7.0 weighted GPA. It was ranked one out of 330 kids in her class to 11 APs and had a 1550 on her SAT. She did track and field, cross country, DECA was on the Ronald McDonald House Teen Advisory Council, was a Board of Education student representative, did track club, student government, SAT peer tutoring, and was treasure of NHS. So does not have a noble prize hurt your chances of getting into top universities despite being valedictorian? Now, this girl was accepted to Harvard, Princeton, and Yale. It is worth noting that she didn't get into Boston University, Stanford, UChicago, or UPEN. But she did get into Brown, Brandeis, Cornell, Dartmouth, Rutgers, UVA, and Wallucyly on top of Harvard, Princeton, and Yale. Two in five for five at the IBs. No need for a noble prize with this one.",
            "This is the college application of a low income Hispanic single family student that had a 3.65 GPA and 1100 SAT. Despite being in a tough situation, the student took the time to take 15 hours of dual credit courses at her local community college. She also had 16 honor bands, state champion and FFA competitions, had numerous FFA degrees, and a certificate of merit from the state government. Namely, she was part of band and choir in high school. She applied to West Point, Naval Academy, Coast Guard, Ole Miss, Southern Mississippi, Delta State, and Nito Wamba Community College, and North East Mississippi Community College. She's accepted to all of them except for West Point and the Naval Academy. She's currently attending Ole Miss, WStore.",
            "These are the universities you get into with a 3.5 unwitted GPA and a 1340 on your SAT. Let's take a look at the rest of her college application and see what college is she got into. So this student is also first gen low income and is from Louisiana. She wants to major in international relations, slash economics. And as for extra curricularity, she was a member of her local children's chorus for two years. Did orchestra, band, chorus was an RA and her dorm she went to a boarding school. She's over schools, black student union, bound her over schools, step team, slash club, volunteered to help out at a local community center assisting in adult literacy classes. This is summer job and a local restaurant for two years. As you know, see I'm not really exceptional at this stuff. I'm just there. I enjoy online activities, which means that in her essay is the authenticity and genuineness of her joy in doing these things is going to shine through, which is very, very important. She also notes here that most of her supplements for her targets are likely objectively bad because she procrastinated and left them to the last minute. And to help with her application. So, she applied to almost 30 universities. Let's go down this list. Rejective from NYU, Rejective from USC, Rejective from UCLA, Weightless from UCSB, Weightless from UCST, Dective from UC Berkeley, Accepted to Scripts, Accepted to Boss University, Weightless from American U, Accepted to George Washington, Weightless from Syracuse and Skidmore, Dective from Barnard, Northeastern and Bard, Accepted to Penn State, SLU Madrid, UMass Amherst, and the University of Arizona. Rejective from Sciences Po, Dective to Spellman College, Weightless from Howard, Accepted to Clarkson and Accepted to the New School. She is committed to Boston University, still very, very solid school. See? We're sound in the end.",
            "This valetorn from a small town in Oklahoma got into Columbia last year. Here's our full college application. She had a 4.0 GPA but a 1240 SAT which is a little bit on the lower side. Her school didn't run for many APs so she took the one that was offered to her. AP Lang. She was part of Oklahoma High Honor Society, Oklahoma Indian Honor Society and NHS. She won a fully funded opportunity to study in Germany from the US Department of State who wasn't able to go because of COVID. She also created a podcast that won an award from Apple and Spotify. She was the first in her school's history to go to an Ivy League. Her town has 3,000 people by the way. Her essay was about meeting an older woman where she writes 79 year old woman approached me as I was unloading the truck. Gleening Twinkle in her eye was an indication of a warmth. She could give me a warm plate of fresh snickered little cookies that she introduced herself as our neighbor. Living in the department right next door. The Arlene ended up being the light in the dark that I needed to help me through a rough patch in my life. Authentic and sweet. Those are the two ingredients you need for a group college essay. Anyway, she applied through Quest Bridge and got into her first choice, Columbia. From a small town of 3,000 Oklahoma to the big city. This is the dream.",
            "Here is the college application of a valedictorian that got into Yale. 104.1 out of 100 GPA. 1520 SAT sport. Took a bunch of AP courses. Human Geo. World History, US History, English Lang, Physics, Chemistry, European History, Psychology, Literature, Latin, Calc 3, Art History, Macro, US Government, and Biology. He was a marketing intern for four years at a non-profit organization, worked with publications like Scholastic and Bloomsburg, class president, volunteer at an archaeological dig site and archived for two years. Volunteer for an art museum did research at a prestigious scholarship program at UT Austin on Hellistic Healthcare, ready for a global health student run magazine, graphic designer, and was president and historian of Hoseph. He applied early action to Yale and after he got in, he was like, eh, not only need to apply anywhere else. This is a type of application that I want you folks to be inspired by. He completely did what he was interested in. He worked with Scholastic, volunteered at an archaeological dig site, art museum, holistic healthcare, global health. You do what interests you and then you tie the thread that brings everything together. That's the key.",
            "Here's what happens when you apply to Harvard with a 2.5 GPA and a thousand SAT score. As for some other information, he's white and high income. He's soon intended to major in sports science and as for his awards, he was voted best athlete at his school. He played varsity football for four years. He was triple-captain for two of those four years. Coached football to young kids. He applied to just four universities. Harvard, Yale, Flandersmith College and the University of Miami. Did you make your guess as to whether he got in or rejected? Well, he got rejected from Harvard and yet. He was also rejected from you, Miami, but accepted to Flandersmith College. So there you go, one for four. Now this isn't to say that I haven't seen kids with like 2.0 GPAs get to Harvard. It's possible. As long as you're able to donate a small sum of $100 million written out to Harvard College, they theoretically are happy to invite you to be a part of the Harvard family in the name of prosperity and continued donations. So the game is very simple, ladies and gentlemen. Just make $100 million. I don't care how you do it. You could go full Walter White, collect the cash and make the dash. And you're in. All you gotta do. Clearly this kid didn't make $100 million, so he didn't have the right idea. But you know, it's alright. Flandersmith it is.",
            "Here's how you get into MIT with perhaps mediocre stats as an Asian. This gets said to himself, he didn't have the most impressive stats and extracurriculars. He wants me to share a story and keep him anonymous to share the other side of competitive college admissions. So, one hour, let's take a look. He had a 1500 SAT and a 5.25 weighted GPA who was ranked 8 out of 575 kids in his class. But to be fair, he did have a good point. He is an Asian male. And as for his extracurriculars, well, he played soccer. He has a tutoring company that he ran for three years. He made 15 grand, pretty solid. He had a power washing company that he ran for four years and made 13 grand. And he was part of Science Bowl. As for his MIT essays, well, they were actually pretty interesting. His essay about why MIT is interesting to him, it reads like this. I discovered my passion for business, my freshman year after founding scholarly solutions tutoring company. I found satisfaction in victorious texts for my longtime student, Jose, and the smiles at the Montgomery County Animal Shelter when we increase their housing supply by donating half of our profits. This inspired me to start power washing plus, which provides underserved homes with three proper cleanings. I've witnessed through my businesses that impact I can make through acts of kindness. I'm not just an entrepreneur, I'm a difference maker. And I'm excited to collaborate with like-minded people at MIT to better our world. He intends to major in business at MIT. Obviously, this checks out. Well, what does this mean? You need a good GPA and you need a pretty solid SAT score. And if you need help with your SAT acelete.ai, the best tool to improve your SAT score. Just saying.",
            "His high school kid is delusional. He says his academics and extracurriculars are below average for MIT where he got in and he is just insanely thankful for being matched with MIT despite his stats. Let me show you what his stats were. He had a 4.0 out of 4.0 unwitted GPA, 5.57 weighted out of 5.0 GPA. He was ranked 15 out of 491 kids in his class and he took 8 AP courses. He's a low income first gen Asian student that applied through questbridge. And as for his extracurriculars, he designed an auger brick extruder for a lunar rover. It's an independent study with NASA in 12th grade where he worked with two NASA engineers as a group of three and he's working on divider covers used for the ISS. From the date that he sent me his college application here, he says that the project finalizes in three weeks, gets sent out for production and review by NASA and then literally flies out to the ISS. He had a code ninja's internship as a programming instructor. It's self started a business during COVID where he troubleshoot into fixed hundreds of computers through call and live chat generated $20,000 which by the way he used to keep his family afloat while his dad lost his job during COVID. And a couple light other projects like co-creating an algorithm for decrypting the vignier cypher without a key applying statistical analysis of Friedman test and Kirk Kost's principle which I have absolutely no idea what the hell any of these things are. But anyway, he was accepted to MIT which seems like a very fitting college for the students. Now to say that the student got lucky and is below average for a school like MIT is a testament to how insane things are nowadays. But don't lose hope. Just build an object that goes out into space while you work alongside NASA engineers and maybe you will also get insanely lucky like this student who, you know, according to him is below average and you'll get into MIT. MIT. My kid.",
             "Here's the college application of a student who got into UPEN this year. He had a 3.9 unweighted GPA and a 1560 on his SAT. He's an international student. He's from Canada. He's Asian and he comes from a high income family. But he's adopted. The first extracurricular he did, well he found a food delivery company that connects students and schools with local restaurants and the business is valued at 350 grand right now. He also created a digital platform for local artisans where he onboarded 35 plus businesses, led a volunteer team, organized workshops and gained recognition from the mayor of his town. He also founded his school's entrepreneurship club, Tended Yale Young Global Scholars. As for his essay, well it starts like this. The day I learned I was adopted, my father handed me a compass, not the digital kind of with a GPS but an old brass one. Whether it's slightly dented, it's needle slightly off center but still pointing forward. He used this as a metaphor for the rest of his essay, where he essentially says that his life is akin to something that is not a map. Rather, life is something to be found. Discovered. He's at UPEN."
        # ... (rest of your sample texts)
    ]
    texts = sample_texts
    labels = ["she/her", "he/him", "he/him", "he/him", "she/her", 
              "he/him", "she/her", "he/him", "she/her",
              "he/him", "she/her", "she/her", "she/her", "she/her",
              "she/her", "she/her", "she/her", "he/him", "he/him","he/him","he/him","he/him"]
    
    # Create pronoun mappings
    pronoun_to_idx, idx_to_pronoun = create_pronoun_mappings(texts, labels)
    save_pronoun_mappings(pronoun_to_idx, idx_to_pronoun, output_dir)
    
    # Initialize tokenizer and save it
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    
    # Split data
    train_texts, val_texts, test_texts = prepare_train_val_test_split(texts)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_texts, tokenizer, pronoun_to_idx, is_training=True, max_length=max_length)
    val_dataset = prepare_dataset(val_texts, tokenizer, pronoun_to_idx, is_training=True, max_length=max_length)
    test_dataset = prepare_dataset(test_texts, tokenizer, pronoun_to_idx, is_training=True, max_length=max_length)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(pronoun_to_idx),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Train model
    trainer = PronounClassifierTrainer(model, device, output_dir)
    model, training_stats = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataloader, desc="Testing")
    print(f"\nAverage test loss: {test_results['loss']:.4f}")
    print("\nTest classification report:")
    print(test_results['classification_report'])
    
    # Save test results
    test_results_path = os.path.join(output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Example usage of the trained model
    example_texts = [
        "How to get into Columbia in under a minute. Here we have the college application of a student who got into Columbia this year. 4.2 GPA, class rank three out of 75 kids. Well, 10 SAT but did not submit. Majoring in international relations, co-founder and president of Model UN at our school, president of Environmental Club, secretary of Arabic, honor society, blood drive captain, varsity volleyball captain, tutored mainly Arabic and international relations. Participated in the Black and Brown Unity panel as a panelist describing experiences with racism in her local community district. And gave a presentation during CSW, which is a committee on the status of women on intersectionality for both local and international viewers. She was the first in her school's history so far to get into an Ivy League. And she was accepted ED to Columbia. Asked for a snippet of her essay, here's her first sentence. When people see me now, they might struggle envisioning a one star-eyed girl who dreamt of nothing more than becoming a writer. There you have it. Columbia in under a one minute. This is just one example though. Keep that in mind.",
        "What the actual is this college application. And before you say it's fake, I found him on LinkedIn. All of it is true. In high school, this kid was the CEO of a company that was valued at $1.5 million. This company was an AI-based, equitable, legal, matchmaking service for all. Which by the way, on LinkedIn, I saw that they raised another $500,000 for this company. And this kid is no longer at college because he's working full-time on this company. He also founded a stem lab for low income students serving about 150 K-thru-six kids raised $110,000 in grants. Was flown out to DC to speak with the government about Indiana education and entrepreneurship. Not to mention he did all the other stuff, like, cat-to-the-clock trial, or international club, summer school for two years. Kids crashed. And a bit of his college essay reads, in the week following the incident, I discovered that some of my classmates had recorded the entire encounter and showed it to the principal, which ultimately led to the teacher's firing. My grief celebration as a hero could possibly happen there. Anyway, he didn't let me know what universities he got into, but he was currently attending the University of Alabama before he dropped out. But he did apply to all the top schools. My team Yale Stanford, Harvard, Princeton, Columbia, and you can go west and final you Chicago, UC Berkeley, Notre Dame, Carnegie Mellon. And yes, he had a 3.8 GPA, 1570 SAT, and double double double block. He did wish he locked in earlier first GPA, because it is on the lower side for top universities. But he also did attend the second best high school, going to US News.",
        "Here's what this girl did to get it to Columbia this year. Okay, Class of 2029, 4.15 GPA, 1540 SAT. Here's her demographic information. Intended to major in theater? As for her awards, nothing too crazy, really. Here are her extracurriculars. Directed a full-length version of Little Woman. Actor for 12 plays, acapella club, technical theater, best buddy secretary, student volunteers, and more, and a couple other things. Nothing ridiculous by any means. Here's a snippet of her essay, if you wanted to take a read real quick. It's pretty solid. And here's her message. The main point is that she didn't focus on college at all senior year in which you try harder for that SAT. Just having fun was able to sell herself as a fun and driven person without winning insane competitions and awards. Accepted, ED to Columbia. There you go. Not an insane, noble prize winning application."
        # ... (rest of your example texts)
    ]
    
    # Load the best model for inference
    best_model = BertForSequenceClassification.from_pretrained(
        os.path.join(output_dir, "best_model")
    )
    best_model.to(device)
    
    # Make predictions
    predictions = predict_pronouns(
        texts=example_texts,
        model=best_model,
        tokenizer=tokenizer,
        idx_to_pronoun=idx_to_pronoun,
        device=device,
        max_length=max_length
    )
    
    # Print predictions
    print("\nPredictions for example texts:")
    for text, pred in zip(example_texts, predictions):
        print(f"\nText snippet: {text[:100]}...")
        print(f"Predicted primary pronoun: {pred}")
        print("-" * 80)

if __name__ == "__main__":
    main()