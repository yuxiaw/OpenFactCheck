import json
from core import register_solver, FactCheckerState, StandardTaskSolver

# Factool example.py category=kbqa response protocol
'''
{
 'average_claim_level_factuality': 0.0,
 'average_response_level_factuality': 0.0,
 'detailed_information': [
    {
      'prompt': 'Introduce Graham Neubig',
      'response': 'Graham Neubig is a professor at MIT',
      'category': 'kbqa',
      'claims': [
        {
            'claim': 'Graham Neubig is a professor at MIT'
        }
      ],
      'queries': [
        [ 'Is Graham Neubig a professor at MIT?', 'Graham Neubig professorship' ]
      ],
      'evidences': [
        {
            'evidence': [ 'I am an Associate Professor at the Carnegie Mellon University Language Technology Institute in the School of Computer Science, and work with a bunch of great ...', 'Missing: MIT? | Show results with:MIT?', 'EI Seminar - Graham Neubig - Learning to Explain and ...', 'Duration: 57:54', 'Posted: Feb 17, 2023', 'I am an Associate Professor at the Carnegie Mellon University Language Technology Institute in the School of Computer Science, and work with a bunch of great ...', 'My research is concerned with language and its role in human communication. In particular, my long-term research goal is to break down barriers in human-human ...', 'Graham Neubig. Associate Professor. Research Interests: Machine Translation · Natural Language Processing · Spoken Language Processing · Machine Learning. My ...', "I am an Associate Professor of Computer Science at Carnegie Mellon University and CEO of… | Learn more about Graham Neubig's work experience, education, ...", 'Graham Neubig received the B.E. degree from the University of Illinois, Urbana ... He is currently an Assistant Professor with Carnegie Mellon University ...' ],
            'source': [ 'http://www.phontron.com/', 'http://www.phontron.com/', 'https://youtube.com/watch?v=CtcP5bvODzY', 'https://youtube.com/watch?v=CtcP5bvODzY', 'https://youtube.com/watch?v=CtcP5bvODzY', 'http://www.phontron.com/', 'https://www.phontron.com/research.php', 'https://lti.cs.cmu.edu/people/222217661/graham-neubig', 'https://www.linkedin.com/in/graham-neubig-10b41616b', 'https://ieeexplore.ieee.org/author/37591106000' ]
        }
      ],
      'claim_level_factuality': [
        {
            'reasoning': 'The given text is non-factual. Multiple pieces of evidence indicate that Graham Neubig is an Associate Professor at the Carnegie Mellon University Language Technology Institute in the School of Computer Science, not at MIT.',
            'error': 'Graham Neubig is not a professor at MIT.',
            'correction': 'Graham Neubig is a professor at Carnegie Mellon University.',
            'factuality': False,
            'claim': 'Graham Neubig is a professor at MIT'
        }
      ],
      'response_level_factuality': False
    }
  ]
}
'''

##
#
# Factool Data Post-Editor
#
# Notes:
#   Factool response post-processor. Used to presents the results in human-readable format and to save the analysis in a JSON file.
#
##
@register_solver("factool_blackbox_post_editor", "claim_info", "claim_info")
class FactoolBlackboxPostEditor(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.path_save_analysis = args.get("path_save_analysis","factool_evidence_analysis.json")

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claim_info = state.get(self.input_name)

        # Restructure some of the output for concatenation (corrected claims)
        edited_claims = ''
        for clf in claim_info['detailed_information'][0]['claim_level_factuality']:
            edited_claims += 'Claim: "' + clf['claim'] + '" => '
            edited_claims += ('' if (clf['error'] == 'None' or len(clf['error']) == 0) else (clf['error'] + ' '))
            edited_claims += ('' if (clf['reasoning'] == 'None' or len(clf['reasoning']) == 0) else clf['reasoning'])
            edited_claims += ((' ' + clf['claim']) if (clf['correction'] == 'None' or len(clf['correction']) == 0) else (' ' + clf['correction']))
            edited_claims += '\n'
        edited_claims = edited_claims[:-1]
        new_claim_info = {}
        new_claim_info[claim_info['detailed_information'][0]['response']] = {
            "edited_claims": edited_claims
        }

        # Serializing json
        json_object = json.dumps(claim_info, indent=4)

        # Writing to sample.json
        with open(self.path_save_analysis, "w") as outfile:
            outfile.write(json_object)

        state.set(self.output_name, new_claim_info)
        return True, state
