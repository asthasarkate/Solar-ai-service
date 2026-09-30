import os

f = 'D:/backend/src/models/Prediction.js'
c = open(f).read()
old_image = '    imageUrl: {\n      type: String,\n      required: true,\n    },'
new_image = '    diyGuidance: {\n      type: String,\n      required: true,\n    },\n    diySafe: {\n      type: Boolean,\n      required: true,\n    },\n' + old_image
if 'diyGuidance:' not in c:
    c = c.replace(old_image, new_image)
    open(f, 'w').write(c)
    print('Prediction.js updated')

f2 = 'D:/backend/src/controllers/predictionController.js'
c2 = open(f2).read()
if 'diyGuidance: result.diyGuidance,' not in c2:
    c2 = c2.replace('      recommendedProfessional: result.recommendedProfessional,\n      imageUrl,', '      recommendedProfessional: result.recommendedProfessional,\n      diyGuidance: result.diyGuidance,\n      diySafe: result.diySafe,\n      imageUrl,')
    c2 = c2.replace('      recommendedProfessional: prediction.recommendedProfessional,\n      imageUrl: prediction.imageUrl,', '      recommendedProfessional: prediction.recommendedProfessional,\n      diyGuidance: prediction.diyGuidance,\n      diySafe: prediction.diySafe,\n      imageUrl: prediction.imageUrl,')
    open(f2, 'w').write(c2)
    print('predictionController.js updated')

f3 = 'D:/backend/src/services/aiService.js'
c3 = open(f3).read()
if 'diyGuidance, diySafe' not in c3:
    c3 = c3.replace('const { faultType, severity, confidence, recommendation, recommendedProfessional } = response.data;', 'const { faultType, severity, confidence, recommendation, recommendedProfessional, diyGuidance, diySafe } = response.data;')
    c3 = c3.replace('    return { faultType, severity, confidence, recommendation, recommendedProfessional };', '    return { faultType, severity, confidence, recommendation, recommendedProfessional, diyGuidance, diySafe };')
    
    old_mock = '  return { faultType, severity, confidence, recommendation, recommendedProfessional };\n};'
    new_mock = """  let diySafe = false;
  let diyGuidance = "";
  if (faultType === 'Dust') {
    diySafe = true;
    diyGuidance = "Safe to clean yourself with water and a soft brush.";
  } else if (faultType === 'Shading') {
    diySafe = true;
    diyGuidance = "If caused by vegetation, trimming is safe to do yourself.";
  } else if (faultType === 'Cracks') {
    diySafe = false;
    diyGuidance = "Not safe for DIY -- contact a certified technician.";
  } else if (faultType === 'Physical Damage') {
    diySafe = false;
    diyGuidance = "Not safe for DIY -- contact a professional.";
  }
  return { faultType, severity, confidence, recommendation, recommendedProfessional, diyGuidance, diySafe };
};"""
    c3 = c3.replace(old_mock, new_mock)
    open(f3, 'w').write(c3)
    print('aiService.js updated')
